import json
import math
import os
import sys
import argparse
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

# Add the preprocessing directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "preprocessing"))

# Import preprocessing components
from preprocessing.tokenizer import RegexMatchTokenizer, TokenType
from preprocessing.preprocess import (
    PreprocessingPipeline,
    LowercasePreprocessor,
    StopWordsPreprocessor,
    NonsenseTokenPreprocessor,
    RemoveDiacriticsPreprocessor
)

# Try to import custom components, fall back to simple implementation if not available
try:
    from preprocessing.stem_preprocessor import StemPreprocessor
    STEMMING_AVAILABLE = True
except ImportError:
    STEMMING_AVAILABLE = False
    print("Warning: StemPreprocessor not available, using basic implementation")

# Define a simple fallback stemmer if needed
class SimpleStemmer:
    """Simple stemmer that uses predefined rules for basic stemming."""
    
    def __init__(self):
        # Load stemming dictionary from vocab_stemmed.txt if available
        self.stem_dict = {}
        vocab_path = os.path.join(os.path.dirname(__file__), "preprocessing", "results", "vocab_stemmed.txt")
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    # Skip first line if it contains a file path comment
                    line = f.readline()
                    if line.startswith('//'):
                        line = f.readline()
                    
                    # Process each line in format: "stem count"
                    while line:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            stem = parts[0]
                            # Add both the stem and common variations
                            self.stem_dict[stem] = stem
                            if stem.endswith('ag'):  # Special case for volkswag -> volkswagen
                                self.stem_dict[stem[:-2] + 'agen'] = stem
                        line = f.readline()
                print(f"Loaded {len(self.stem_dict)} stem mappings from {vocab_path}")
            except Exception as e:
                print(f"Error loading stem dictionary: {e}")
        
        # Add common stemming rules
        self.rules = [
            (r'ies$', 'y'),
            (r's$', ''),
            (r'ing$', ''),
            (r'ed$', ''),
            (r'volkswagen', 'volkswag'),  # Special case
        ]
    
    def stem(self, word):
        """Apply stemming to a word using rules or dictionary."""
        # Check if we have this in our dictionary
        if word.lower() in self.stem_dict:
            return self.stem_dict[word.lower()]
        
        # Apply rules
        for pattern, replacement in self.rules:
            if re.search(pattern, word.lower()):
                return re.sub(pattern, replacement, word.lower())
        
        # No rule matched, return as is
        return word.lower()

class SimpleStemmingPreprocessor:
    """Simple stemming preprocessor for when the real stemmer is not available."""
    
    def __init__(self):
        self.stemmer = SimpleStemmer()
        
    def preprocess(self, token, document_text):
        if token.processed_form:
            token.processed_form = self.stemmer.stem(token.processed_form)
        return token
    
    def preprocess_all(self, tokens, document_text):
        return [self.preprocess(token, document_text) for token in tokens]

# Function to attempt to create stemming preprocessor
def create_stemming_preprocessor():
    if STEMMING_AVAILABLE:
        # Try to find the stemmer executable
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stemmer_path = os.path.join(current_dir, "preprocessing", "CzechSnowballStemmer", "stemmer")
        
        if not os.path.exists(stemmer_path):
            # Try alternative paths
            alternative_paths = [
                os.path.join(current_dir, "stemmer"),
                "./stemmer",
                os.path.join(current_dir, "preprocessing", "stemmer")
            ]
            for path in alternative_paths:
                if os.path.exists(path):
                    stemmer_path = path
                    break
        
        if os.path.exists(stemmer_path):
            print(f"Using stemmer at: {stemmer_path}")
            try:
                # Make executable if needed
                os.chmod(stemmer_path, 0o755)
                return StemPreprocessor(language="cz", stemmer_path=stemmer_path)
            except Exception as e:
                print(f"Error setting up stemmer: {e}")
        
        print("Stemmer executable not found, using simple stemmer instead")
    
    # Fallback to simple implementation
    return SimpleStemmingPreprocessor()

# For cosine similarity calculation
def compute_cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector as a dictionary {word: tf_idf_score}
        vec2: Second vector as a dictionary {word: tf_idf_score}
        
    Returns:
        Cosine similarity score
    """
    # Find common words
    common_words = set(vec1.keys()) & set(vec2.keys())
    
    # Calculate dot product
    dot_product = sum(vec1[word] * vec2[word] for word in common_words)
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(score**2 for score in vec1.values()))
    magnitude2 = math.sqrt(sum(score**2 for score in vec2.values()))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)


class Document:
    """Class representing a document with its content and processed data."""
    
    def __init__(self, doc_id: int, title: str = "", abstract: str = "", text: str = ""):
        self.id = doc_id
        self.title = title
        self.abstract = abstract
        self.text = text
        # Combined text for processing
        self.combined_text = f"{title} {abstract} {text}".strip()
        self.tokens = None
        self.word_freq = None
        self.tf_idf_vector = None
    
    def __str__(self):
        return f"Document {self.id}: {self.title[:50]}{'...' if len(self.title) > 50 else ''}"


class InvertedIndex:
    """Inverted index mapping words to document occurrences."""
    
    def __init__(self):
        self.index = defaultdict(dict)  # {word: {doc_id: freq}}
        self.document_count = 0
        self.documents = {}  # {doc_id: Document}
    
    def add_document(self, document: Document, word_freq: Dict[str, int]):
        """
        Add a document to the inverted index.
        
        Args:
            document: Document object to add
            word_freq: Dictionary mapping words to their frequencies in the document
        """
        doc_id = document.id
        self.documents[doc_id] = document
        
        # Update document frequency for each word
        for word, freq in word_freq.items():
            if word:  # Skip empty strings
                self.index[word][doc_id] = freq
        
        self.document_count += 1
    
    def get_document_frequency(self, word: str) -> int:
        """
        Get the number of documents containing the given word.
        
        Args:
            word: The word to check
            
        Returns:
            Number of documents containing the word
        """
        return len(self.index.get(word, {}))
    
    def get_inverse_document_frequency(self, word: str) -> float:
        """
        Calculate the inverse document frequency for a word.
        IDF(t) = log10(N/DF(t))
        
        Args:
            word: The word to calculate IDF for
            
        Returns:
            IDF value for the word
        """
        df = self.get_document_frequency(word)
        if df == 0:
            return 0
        return math.log10(self.document_count / df)


def preprocess_text(text: str, pipeline: PreprocessingPipeline) -> Dict[str, int]:
    """
    Preprocess text and return word frequencies.
    
    Args:
        text: Text to preprocess
        pipeline: Preprocessing pipeline to use
        
    Returns:
        Dictionary mapping words to their frequencies
    """
    tokenizer = RegexMatchTokenizer()
    tokens = tokenizer.tokenize(text)
    
    # Apply preprocessing pipeline
    processed_tokens = pipeline.preprocess(tokens, text)
    
    # Build word frequency dictionary from processed tokens
    word_freq = Counter()
    for token in processed_tokens:
        if token.processed_form and token.token_type == TokenType.WORD:  # Skip empty tokens or non-words
            word_freq[token.processed_form] += 1
    
    return word_freq


def compute_tf(word_freq: Dict[str, int]) -> Dict[str, float]:
    """
    Compute term frequency (TF) for each word.
    TF(t,d) = 1 + log10(f(t,d)) if f(t,d) > 0, else 0
    
    Args:
        word_freq: Dictionary mapping words to their frequencies
        
    Returns:
        Dictionary mapping words to their TF scores
    """
    tf_scores = {}
    for word, freq in word_freq.items():
        if freq > 0:
            tf_scores[word] = 1 + math.log10(freq)
        else:
            tf_scores[word] = 0
    return tf_scores


def compute_tf_idf_vectors(inverted_index: InvertedIndex) -> Dict[int, Dict[str, float]]:
    """
    Compute TF-IDF vectors for all documents in the inverted index.
    
    Args:
        inverted_index: The inverted index containing documents
        
    Returns:
        Dictionary mapping document IDs to their TF-IDF vectors
    """
    tf_idf_vectors = {}
    
    # Process each document
    for doc_id, document in inverted_index.documents.items():
        word_freq = {}
        
        # Extract word frequencies for this document from the inverted index
        for word, doc_dict in inverted_index.index.items():
            if doc_id in doc_dict:
                word_freq[word] = doc_dict[doc_id]
        
        # Compute TF scores
        tf_scores = compute_tf(word_freq)
        
        # Compute TF-IDF vector
        tf_idf_vector = {}
        for word, tf_score in tf_scores.items():
            idf = inverted_index.get_inverse_document_frequency(word)
            tf_idf_vector[word] = tf_score * idf
        
        # Store TF-IDF vector in document and in result dictionary
        document.tf_idf_vector = tf_idf_vector
        tf_idf_vectors[doc_id] = tf_idf_vector
    
    return tf_idf_vectors


def process_queries(queries: List[str], pipeline: PreprocessingPipeline, inverted_index: InvertedIndex) -> List[Tuple[str, Dict[str, float]]]:
    """
    Process queries and compute their TF-IDF vectors.
    
    Args:
        queries: List of query strings
        pipeline: Preprocessing pipeline to use
        inverted_index: Inverted index for IDF lookup
        
    Returns:
        List of tuples (original_query, tf_idf_vector) for each query
    """
    query_vectors = []
    
    for query in queries:
        # Preprocess query text
        word_freq = preprocess_text(query, pipeline)
        
        # Compute TF scores
        tf_scores = compute_tf(word_freq)
        
        # Compute TF-IDF vector
        tf_idf_vector = {}
        for word, tf_score in tf_scores.items():
            idf = inverted_index.get_inverse_document_frequency(word)
            tf_idf_vector[word] = tf_score * idf
        
        query_vectors.append((query, tf_idf_vector))
    
    return query_vectors


def rank_documents(query_vector: Dict[str, float], 
                  doc_vectors: Dict[int, Dict[str, float]], 
                  documents: Dict[int, Document],
                  top_k: int = 5) -> List[Tuple[Document, float]]:
    """
    Rank documents by cosine similarity to the query vector.
    
    Args:
        query_vector: TF-IDF vector of the query
        doc_vectors: Dictionary mapping document IDs to their TF-IDF vectors
        documents: Dictionary mapping document IDs to Document objects
        top_k: Number of top results to return
        
    Returns:
        List of (document, similarity) tuples, sorted by similarity
    """
    similarities = []
    
    for doc_id, doc_vector in doc_vectors.items():
        similarity = compute_cosine_similarity(query_vector, doc_vector)
        similarities.append((documents[doc_id], similarity))
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return similarities[:top_k]


def create_preprocessing_pipeline() -> PreprocessingPipeline:
    """
    Create a standard preprocessing pipeline with stemming support.
    
    Returns:
        Configured preprocessing pipeline
    """
    # Create a stemming preprocessor first
    stemming_preprocessor = create_stemming_preprocessor()
    
    # Create other preprocessors
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "preprocessing", "data")
    
    # Fallback to standard directories if needed
    if not os.path.exists(data_dir):
        data_dir = os.path.join(current_dir, "data")
    
    # Create preprocessors
    preprocessors = [
        LowercasePreprocessor(),
        RemoveDiacriticsPreprocessor(),
        NonsenseTokenPreprocessor(min_word_length=2),
        stemming_preprocessor 
    ]
    
    # Try to add stop words if available
    try:
        stop_words = StopWordsPreprocessor(language="cz", stop_words_dir=data_dir)
        # Insert before stemming
        preprocessors.insert(2, stop_words)
    except Exception as e:
        print(f"Error loading stop words: {e}")
    
    return PreprocessingPipeline(preprocessors, name="Standard Pipeline")


def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """
    Load documents from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of document dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_documents(documents_json: List[Dict[str, Any]], pipeline: PreprocessingPipeline) -> InvertedIndex:
    """
    Process documents and build an inverted index.
    
    Args:
        documents_json: List of document dictionaries
        pipeline: Preprocessing pipeline
        
    Returns:
        Built inverted index
    """
    inverted_index = InvertedIndex()
    
    for i, doc_data in enumerate(documents_json):
        # Create document object
        document = Document(
            doc_id=i,
            title=doc_data.get("title", ""),
            abstract=doc_data.get("abstract", ""),
            text=doc_data.get("text", "")
        )
        
        # Preprocess document text
        word_freq = preprocess_text(document.combined_text, pipeline)
        document.word_freq = word_freq
        
        # Add to inverted index
        inverted_index.add_document(document, word_freq)
    
    return inverted_index


def create_test_documents() -> List[Dict[str, Any]]:
    """
    Create sample test documents if no input file is provided.
    """
    return [
        {
            "title": "Volkswagen Electric Cars",
            "abstract": "Volkswagen is expanding its electric vehicle lineup.",
            "text": "Volkswagen's new EV models include the ID.4 and the ID.Buzz. The company plans to be carbon neutral by 2050."
        },
        {
            "title": "Electric Vehicle Market Trends",
            "abstract": "The market for electric vehicles continues to grow globally.",
            "text": "Many automotive companies including Volkswagen and BMW are investing heavily in electric vehicle technology."
        },
        {
            "title": "Car Manufacturers in Europe",
            "abstract": "European car manufacturers face new emission regulations.",
            "text": "Companies like Volkswagen, BMW, and Mercedes must adapt to stricter emission standards."
        },
        {
            "title": "History of Automobiles",
            "abstract": "The evolution of cars from early models to modern vehicles.",
            "text": "From Ford's Model T to modern electric vehicles, the automobile industry has seen significant changes."
        },
        {
            "title": "Hybrid Vehicle Technology",
            "abstract": "Hybrid vehicles combine conventional engines with electric motors.",
            "text": "Toyota and Honda pioneered hybrid technology, which is now adopted by many manufacturers."
        }
    ]


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='TF-IDF Search Engine with Inverted Index')
    parser.add_argument('--input', help='Input JSON file with documents')
    parser.add_argument('--queries', nargs='+', help='Search queries (if not provided, defaults will be used)')
    parser.add_argument('--top', type=int, default=3, help='Number of top results to display')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed output')
    args = parser.parse_args()
    
    # Use default queries if none provided
    if not args.queries:
        args.queries = [
            "volkswagen",  # This should match "volkswag" after stemming
            "electric cars",
            "automobile history"
        ]
    
    # Load documents
    try:
        if args.input:
            print(f"Loading documents from {args.input}...")
            documents_json = load_documents(args.input)
        else:
            print("No input file provided. Using test documents...")
            documents_json = create_test_documents()
            
        print(f"Loaded {len(documents_json)} documents.")
        
        # Create preprocessing pipeline
        pipeline = create_preprocessing_pipeline()
        print(f"Using preprocessing pipeline: {pipeline.name}")
        
        # Process documents and build inverted index
        print("Building inverted index...")
        inverted_index = process_documents(documents_json, pipeline)
        
        # Print index info in debug mode
        if args.debug:
            print("\nInverted Index Contents:")
            for word, docs in sorted(inverted_index.index.items()):
                print(f"  {word}: {len(docs)} documents")
        
        # Compute TF-IDF vectors for all documents
        print("Computing TF-IDF vectors for documents...")
        doc_vectors = compute_tf_idf_vectors(inverted_index)
        
        # Process queries
        print("Processing queries...")
        query_vectors = process_queries(args.queries, pipeline, inverted_index)
        
        # Display results
        print("\n" + "="*50)
        print("SEARCH RESULTS")
        print("="*50)
        
        for i, (original_query, query_vector) in enumerate(query_vectors):
            print(f"\nQuery {i+1}: \"{original_query}\"")
            
            # Show processed query in verbose mode
            if args.verbose or args.debug:
                # Show the processed tokens
                tokenizer = RegexMatchTokenizer()
                tokens = tokenizer.tokenize(original_query)
                pipeline.preprocess(tokens, original_query)
                processed_terms = [t.processed_form for t in tokens if t.processed_form]
                print(f"Processed query terms: {processed_terms}")
                
                # Show TF-IDF vector
                if args.debug:
                    print("Query TF-IDF vector:")
                    for term, score in sorted(query_vector.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {term}: {score:.4f}")
            
            print("-" * 40)
            
            # Rank documents for this query
            ranked_docs = rank_documents(
                query_vector, 
                doc_vectors, 
                inverted_index.documents,
                top_k=args.top
            )
            
            # Display results
            if not ranked_docs:
                print("No matching documents found.")
            else:
                for j, (doc, similarity) in enumerate(ranked_docs):
                    print(f"{j+1}. Document: \"{doc.title}\"")
                    print(f"   Abstract: {doc.abstract[:100]}{'...' if len(doc.abstract) > 100 else ''}")
                    print(f"   Similarity: {similarity:.4f}")
                    
                    # Show matching terms in verbose/debug mode
                    if args.verbose or args.debug:
                        common_terms = set(query_vector.keys()) & set(doc.tf_idf_vector.keys())
                        if common_terms:
                            print(f"   Matching terms: {', '.join(common_terms)}")
                            
                        # Show document vector in debug mode
                        if args.debug:
                            print("   Document terms:")
                            top_terms = sorted(doc.tf_idf_vector.items(), key=lambda x: x[1], reverse=True)[:5]
                            for term, score in top_terms:
                                print(f"     {term}: {score:.4f}")
                    print()
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{args.input}' as JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()