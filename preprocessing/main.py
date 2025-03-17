import json
import argparse
from typing import Iterable, Dict
import os
import time
from collections import Counter

from tokenizer import RegexMatchTokenizer, Tokenizer, TokenType
from preprocess import (
    PreprocessingPipeline, 
    LowercasePreprocessor, 
    StopWordsPreprocessor, 
    NonsenseTokenPreprocessor,
    RemoveDiacriticsPreprocessor
)

# Import stemming for automatic execution
import stemming

# Import lemmatization
import lemmatization

class Document:
    def __init__(self, text: str, title: str = "", abstract: str = ""):
        self.title = title
        self.abstract = abstract
        self.text = text
        # Combined text for processing
        self.combined_text = f"{title} {abstract} {text}".strip()
        self.tokens = None
        self.vocab = None

    def tokenize(self, tokenizer: Tokenizer=None):
        tokenizer = tokenizer or RegexMatchTokenizer()
        # Tokenize the combined text
        self.tokens = tokenizer.tokenize(self.combined_text)
        return self

    def preprocess(self, preprocessing_pipeline: PreprocessingPipeline):
        preprocessing_pipeline.preprocess(self.tokens, self.combined_text)
        return self

def build_vocabulary(documents: Iterable[Document]):
    vocab = Counter()
    for doc in documents:
        # Count only tokens with non-empty processed_form (filter out empty tokens after stop words removal)
        vocab.update([token.processed_form for token in doc.tokens if token.processed_form])
    return vocab

def write_weighted_vocab(vocab, file):
    """Write vocabulary with occurrence counts."""
    for key, value in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
        file.write(f"{key} {value}\n")

def write_vocab_only(vocab, file, documents):
    """Write vocabulary without counts, only relevant tokens (words).
    
    Filter tokens and write only those of type WORD.
    """
    # Create a dictionary that maps tokens to their types
    token_types = {}
    for doc in documents:
        for token in doc.tokens:
            if token.processed_form:
                token_types[token.processed_form] = token.token_type

    # Write only tokens of type WORD, alphabetically sorted
    for key in sorted(vocab.keys()):
        # If token is not in type mapping or is of type WORD, write it
        if key in token_types and token_types[key] == TokenType.WORD:
            file.write(f"{key}\n")

def process_with_pipeline(documents, pipeline):
    """Process documents with the given pipeline."""
    start_time = time.time()
    
    processed_documents = []
    for doc in documents:
        # Copy original document for new pipeline
        new_doc = Document(text=doc.text, title=doc.title, abstract=doc.abstract)
        new_doc.tokenize().preprocess(pipeline)
        processed_documents.append(new_doc)
    
    # Time measurement
    elapsed_time = time.time() - start_time
    
    # Vocabulary and statistics
    vocab = build_vocabulary(processed_documents)
    
    return {
        'documents': processed_documents,
        'vocabulary': vocab,
        'time': elapsed_time,
        'vocab_size': len(vocab)
    }

def print_pipeline_stats(pipeline_name, stats, original_vocab_size):
    """Print statistics for the given pipeline."""
    print(f"\n--- {pipeline_name} ---")
    print(f"Vocabulary size: {stats['vocab_size']} (reduction of {original_vocab_size - stats['vocab_size']} words, {(original_vocab_size - stats['vocab_size'])/original_vocab_size*100:.1f}%)")
    print(f"Processing time: {stats['time']:.2f} s")
    
    # Print 10 most frequent words
    print("10 most frequent words:")
    for word, count in sorted(stats['vocabulary'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {word}: {count}")

def create_pipeline(config: Dict) -> PreprocessingPipeline:
    """
    Create pipeline based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        PreprocessingPipeline configured according to settings
    """
    preprocessors = []
    preprocessing_config = config["preprocessing"]
    
    # Ideal order: stop_words -> meaningless_token -> lowercase -> diacritics -> stemming
    
    # 1. Stop words
    stop_words_config = preprocessing_config["stop_words"]
    if stop_words_config["use"] and stop_words_config["language"] != "none":
        preprocessors.append(StopWordsPreprocessor(language=stop_words_config["language"]))
    
    # 2. Remove meaningless tokens
    nonsense_config = preprocessing_config["nonsense_tokens"]
    if nonsense_config["remove"]:
        preprocessors.append(NonsenseTokenPreprocessor(min_word_length=nonsense_config["min_word_length"]))
    
    # 3. Convert to lowercase
    if preprocessing_config["lowercase"]:
        preprocessors.append(LowercasePreprocessor())
    
    # 4. Remove diacritics
    if preprocessing_config["remove_diacritics"]:
        preprocessors.append(RemoveDiacriticsPreprocessor())
    
    # Create pipeline name
    pipeline_parts = []
    if stop_words_config["use"] and stop_words_config["language"] != "none":
        pipeline_parts.append(f"{stop_words_config['language']} stop words")
    if nonsense_config["remove"]:
        pipeline_parts.append(f"min length {nonsense_config['min_word_length']}")
    if preprocessing_config["lowercase"]:
        pipeline_parts.append("lowercase")
    if preprocessing_config["remove_diacritics"]:
        pipeline_parts.append("no diacritics")
    
    pipeline_name = " + ".join(pipeline_parts)
    
    return PreprocessingPipeline(preprocessors, name=pipeline_name)

def load_config(config_file):
    """Load configuration from file with comments support."""
    if not os.path.exists(config_file):
        # If configuration file doesn't exist, create default configuration
        from create_default_config import create_default_config
        create_default_config()
    
    # Load file content
    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove line comments (lines starting with //)
    lines = content.splitlines()
    filtered_lines = []
    for line in lines:
        # Remove comment from line if exists
        line_without_comment = line.split("//")[0]
        # Add line without comment if not empty
        if line_without_comment.strip():
            filtered_lines.append(line_without_comment)
    
    # Join lines back into one string
    filtered_content = "\n".join(filtered_lines)
    
    # Load JSON
    try:
        config = json.loads(filtered_content)
        return config
    except json.JSONDecodeError as e:
        print(f"Error loading configuration file: {e}")
        print("Creating standard configuration...")
        
        # If loading configuration fails, create standard one
        from create_default_config import create_default_config
        create_default_config("config.standard.json")
        
        # Load standard configuration
        with open("config.standard.json", "r", encoding="utf-8") as f:
            return json.load(f)

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Text preprocessing for NLP')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load documents
    input_file = config['input']['file']
    encoding = config['input']['encoding']
    
    documents = []
    with open(input_file, 'r', encoding=encoding) as f:
        data = json.load(f)
        # Load title, abstract and text from each article
        for article in data:
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            text = article.get("text", "")
            
            # Create document only if it has content
            if title or abstract or text:
                doc = Document(text=text, title=title, abstract=abstract)
                documents.append(doc.tokenize())
    
    print(f"Loaded {len(documents)} documents.")
    
    # Original vocabulary without preprocessing
    original_vocab = build_vocabulary(documents)
    original_vocab_size = len(original_vocab)
    print(f"Original vocabulary size: {original_vocab_size}")
    
    # Save original vocabulary
    original_vocab_file = os.path.join(output_dir, config['output']['original_vocab'])
    with open(original_vocab_file, "w", encoding=encoding) as f:
        write_weighted_vocab(original_vocab, f)
    
    # Create pipeline according to configuration
    pipeline = create_pipeline(config)
    
    # Process documents
    print(f"\nProcessing pipeline: '{pipeline.name}'...")
    results = process_with_pipeline(documents, pipeline)
    print_pipeline_stats(pipeline.name, results, original_vocab_size)
    
    # Save preprocessed vocabulary
    preprocessed_vocab_file = os.path.join(output_dir, config['output']['preprocessed_vocab'])
    with open(preprocessed_vocab_file, "w", encoding=encoding) as f:
        write_weighted_vocab(results['vocabulary'], f)
    
    # Apply stemming if requested
    if config['stemming']['use']:
        stemmed_vocab_file = os.path.join(output_dir, config['output']['stemmed_vocab'])
        print("\nPerforming stemming...")
        
        # Use stemming module directly
        stemming.process_stemming(
            input_file=preprocessed_vocab_file,
            output_file=stemmed_vocab_file,
            language=config['stemming']['language'],
            output_dir=output_dir,
            stemmer_path=config['stemming']['stemmer_path']
        )
        
        # Get stemmed vocabulary size
        stemmed_word_count = 0
        with open(stemmed_vocab_file, 'r', encoding=encoding) as f:
            for line in f:
                stemmed_word_count += 1
        
        stemmed_reduction = (original_vocab_size - stemmed_word_count) / original_vocab_size * 100
        
        print("\n--- Results ---")
        print(f"1. Original vocabulary: {original_vocab_size} words")
        print(f"2. After preprocessing: {results['vocab_size']} words")
        print(f"   - Saved in: {preprocessed_vocab_file}")
        print(f"3. After stemming: {stemmed_word_count} words (reduction of {stemmed_reduction:.1f}%)")
        print(f"   - Saved in: {stemmed_vocab_file}")
        print(f"   - Groups of words with same stem: {os.path.join(output_dir, 'stemming_groups.json')}")
    
    # Apply lemmatization if requested
    if config['lemmatization']['use']:
        lemmatized_vocab_file = os.path.join(output_dir, "vocab_lemmatized.txt")
        print("\nPerforming lemmatization...")
        
        # Determine model path by language
        model_path = None  # Use default model by language
        
        # Use lemmatization module directly
        lemmatization.process_lemmatization(
            input_file=preprocessed_vocab_file,
            output_file=lemmatized_vocab_file,
            language=config['lemmatization']['language'],
            output_dir=output_dir,
            model_path=model_path
        )
        
        # Get lemmatized vocabulary size
        lemmatized_word_count = 0
        with open(lemmatized_vocab_file, 'r', encoding=encoding) as f:
            for line in f:
                lemmatized_word_count += 1
        
        lemmatized_reduction = (original_vocab_size - lemmatized_word_count) / original_vocab_size * 100
        
        print("\n--- Results ---")
        print(f"1. Original vocabulary: {original_vocab_size} words")
        print(f"2. After preprocessing: {results['vocab_size']} words")
        print(f"   - Saved in: {preprocessed_vocab_file}")
        
        if config['stemming']['use']:
            print(f"3. After stemming: {stemmed_word_count} words (reduction of {stemmed_reduction:.1f}%)")
            print(f"   - Saved in: {stemmed_vocab_file}")
            print(f"   - Groups of words with same stem: {os.path.join(output_dir, 'stemming_groups.json')}")
        
        print(f"4. After lemmatization: {lemmatized_word_count} words (reduction of {lemmatized_reduction:.1f}%)")
        print(f"   - Saved in: {lemmatized_vocab_file}")
        print(f"   - Groups of words with same lemma: {os.path.join(output_dir, 'lemmatization_groups.json')}")
    else:
        print("\n--- Results ---")
        print(f"1. Original vocabulary: {original_vocab_size} words")
        print(f"2. After preprocessing: {results['vocab_size']} words")
        print(f"   - Saved in: {preprocessed_vocab_file}")
        
        if config['stemming']['use']:
            print(f"3. After stemming: {stemmed_word_count} words (reduction of {stemmed_reduction:.1f}%)")
            print(f"   - Saved in: {stemmed_vocab_file}")
            print(f"   - Groups of words with same stem: {os.path.join(output_dir, 'stemming_groups.json')}")

if __name__ == '__main__':
    main()