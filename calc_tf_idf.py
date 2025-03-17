import math
import re
from collections import Counter
from typing import List, Dict, Tuple


def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text by converting to lowercase and splitting into words.
    
    Args:
        text: Input text string
        
    Returns:
        List of preprocessed words
    """
    # Convert to lowercase and split by non-alphanumeric characters
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def compute_word_frequencies(documents: List[str]) -> List[Counter]:
    """
    Compute the frequency of each word in each document.
    
    Args:
        documents: List of document strings
        
    Returns:
        List of Counter objects containing word frequencies for each document
    """
    doc_word_freqs = []
    for doc in documents:
        words = preprocess_text(doc)
        word_freqs = Counter(words)
        doc_word_freqs.append(word_freqs)
    return doc_word_freqs


def compute_tf(word_freqs: List[Counter]) -> List[Dict[str, float]]:
    """
    Compute term frequency (TF) for each word in each document.
    TF(t,d) = 1 + log10(f(t,d)) if f(t,d) > 0, else 0
    
    Args:
        word_freqs: List of Counter objects with word frequencies
        
    Returns:
        List of dictionaries containing TF scores for each word in each document
    """
    tf_scores = []
    for freqs in word_freqs:
        tf = {}
        for word, freq in freqs.items():
            if freq > 0:
                tf[word] = 1 + math.log10(freq)
            else:
                tf[word] = 0
        tf_scores.append(tf)
    return tf_scores


def compute_df_idf(doc_word_freqs: List[Counter]) -> Dict[str, float]:
    """
    Compute document frequency (DF) and inverse document frequency (IDF) for each word.
    IDF(t) = log10(N/DF(t)), where N is the total number of documents.
    
    Args:
        doc_word_freqs: List of Counter objects with word frequencies for each document
        
    Returns:
        Dictionary mapping words to their IDF scores
    """
    # Get all unique words across all documents
    all_words = set()
    for word_freq in doc_word_freqs:
        all_words.update(word_freq.keys())
    
    # Compute document frequency for each word
    df = Counter()
    for word in all_words:
        for word_freq in doc_word_freqs:
            if word in word_freq:
                df[word] += 1
    
    # Compute IDF for each word
    n_docs = len(doc_word_freqs)
    idf = {}
    for word, doc_freq in df.items():
        idf[word] = math.log10(n_docs / doc_freq)
    
    return idf


def compute_tf_idf(tf_scores: List[Dict[str, float]], idf_scores: Dict[str, float]) -> List[Dict[str, float]]:
    """
    Compute TF-IDF scores for each word in each document.
    TF-IDF(t,d) = TF(t,d) × IDF(t)
    
    Args:
        tf_scores: List of dictionaries containing TF scores
        idf_scores: Dictionary mapping words to their IDF scores
        
    Returns:
        List of dictionaries containing TF-IDF scores for each word in each document
    """
    tf_idf_scores = []
    for tf in tf_scores:
        tf_idf = {}
        for word, tf_score in tf.items():
            tf_idf[word] = tf_score * idf_scores.get(word, 0)
        tf_idf_scores.append(tf_idf)
    return tf_idf_scores


def compute_cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two vectors.
    cos(θ) = (A · B) / (||A|| * ||B||)
    
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


def calculate_similarities(documents: List[str], queries: List[str]) -> List[List[Tuple[str, str, float]]]:
    """
    Calculate cosine similarities between each query and each document.
    
    Args:
        documents: List of document strings
        queries: List of query strings
        
    Returns:
        List of similarity results for each query
    """
    # Process documents
    doc_word_freqs = compute_word_frequencies(documents)
    doc_tf_scores = compute_tf(doc_word_freqs)
    idf_scores = compute_df_idf(doc_word_freqs)
    doc_tf_idf_scores = compute_tf_idf(doc_tf_scores, idf_scores)
    
    # Process queries
    query_word_freqs = compute_word_frequencies(queries)
    query_tf_scores = compute_tf(query_word_freqs)
    query_tf_idf_scores = compute_tf_idf(query_tf_scores, idf_scores)
    
    # Calculate similarities
    all_similarities = []
    for i, query_tf_idf in enumerate(query_tf_idf_scores):
        similarities = []
        for j, doc_tf_idf in enumerate(doc_tf_idf_scores):
            similarity = compute_cosine_similarity(query_tf_idf, doc_tf_idf)
            similarities.append((queries[i], documents[j], similarity))
        all_similarities.append(similarities)
    
    return all_similarities


def main():
    # Sample documents and queries
    documents = [
        "tropical fish include fish found in tropical environments",
        "fish live in a sea",
        "tropical fish are popular aquarium fish",
        "fish also live in Czechia",
        "Czechia is a country"
    ]
    
    queries = [
        "tropical fish sea",
        "tropical fish"
    ]
    
    # Calculate similarities
    all_similarities = calculate_similarities(documents, queries)
    
    # Print results
    for query_similarities in all_similarities:
        for query, doc, similarity in query_similarities:
            print(f'Query: "{query}" Document: "{doc}" Cosine Similarity: {similarity:.4f}')
        print()


if __name__ == "__main__":
    main()