from abc import ABC, abstractmethod
from tokenizer import Token, TokenType
import json
import os
import unicodedata

class TokenPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, token: Token, document: str) -> Token:
        raise NotImplementedError()

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        return [self.preprocess(token, document) for token in tokens]

class LowercasePreprocessor(TokenPreprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = token.processed_form.lower()
        return token


class StopWordsPreprocessor(TokenPreprocessor):
    """Preprocessor for removing stop words."""
    
    def __init__(self, language="both", stop_words_dir="data"):
        """
        Initialize preprocessor for removing stop words.
        
        Args:
            language: Can be 'czech', 'english' or 'both' to determine which stop words to use
            stop_words_dir: Directory containing stop words files
        """
        self.czech_stop_words = set()
        self.english_stop_words = set()
        
        # Load Czech stop words
        cs_path = os.path.join(stop_words_dir, "stopwords-cs.json")
        if os.path.exists(cs_path):
            with open(cs_path, 'r', encoding='utf-8') as f:
                self.czech_stop_words = set(json.load(f))
        
        # Load English stop words
        en_path = os.path.join(stop_words_dir, "stopwords-en.json")
        if os.path.exists(en_path):
            with open(en_path, 'r', encoding='utf-8') as f:
                self.english_stop_words = set(json.load(f))
        
        # Set active stop words based on selected language
        if language == "cz":
            self.stop_words = self.czech_stop_words
        elif language == "en":
            self.stop_words = self.english_stop_words
        else:  # both
            self.stop_words = self.czech_stop_words.union(self.english_stop_words)
    
    def preprocess(self, token: Token, document: str) -> Token:
        """
        If token is a stop word, replace its processed_form with empty string.
        
        Args:
            token: Token to process
            document: Original document

        Returns:
            Processed token
        """
        # Process only words, not numbers, punctuation, etc.
        if token.token_type == TokenType.WORD and token.processed_form.lower() in self.stop_words:
            token.processed_form = ""
        return token


class NonsenseTokenPreprocessor(TokenPreprocessor):
    """Preprocessor for removing nonsense tokens."""
    
    def __init__(self, min_word_length=2, remove_types=None, preserve_types=None):
        """
        Initialize preprocessor for removing nonsense tokens.
        
        Args:
            min_word_length: Minimum word length (shorter will be removed)
            remove_types: List of token types to remove
            preserve_types: List of token types to preserve (takes precedence over remove_types)
        """
        self.min_word_length = min_word_length
        
        # Default token types to remove (if not specified otherwise)
        self.remove_types = remove_types or [
            TokenType.PUNCT, 
            TokenType.TAG
        ]
        
        # Token types to preserve regardless of their length
        self.preserve_types = preserve_types or [
            TokenType.URL, 
            TokenType.DATE
        ]
        
    def preprocess(self, token: Token, document: str) -> Token:
        """
        Remove nonsense tokens.
        
        Args:
            token: Token to process
            document: Original document

        Returns:
            Processed token
        """
        # Preserve tokens of specific types regardless of length
        if token.token_type in self.preserve_types:
            return token
            
        # Remove tokens of unwanted types
        if token.token_type in self.remove_types:
            token.processed_form = ""
            return token
            
        # Remove words that are too short
        if token.token_type == TokenType.WORD and len(token.processed_form) < self.min_word_length:
            token.processed_form = ""
            return token
            
        # In other cases preserve the token
        return token


class RemoveDiacriticsPreprocessor(TokenPreprocessor):
    """Preprocessor for removing diacritics."""
    
    def preprocess(self, token: Token, document: str) -> Token:
        """
        Remove diacritics from token.
        
        Args:
            token: Token to process
            document: Original document

        Returns:
            Processed token without diacritics
        """
        # NFD normalization - decomposes characters with diacritics into base character and diacritical mark
        # then removes all diacritical marks (category Mn - Mark, nonspacing)
        token.processed_form = ''.join(c for c in unicodedata.normalize('NFD', token.processed_form)
                                    if not unicodedata.combining(c))
        return token


class PreprocessingPipeline:
    def __init__(self, preprocessors: list[TokenPreprocessor], name: str = None):
        self.preprocessors = preprocessors
        self.name = name or "Anonymous pipeline"

    def preprocess(self, tokens: list[Token], document: str) -> list[Token]:
        for preprocessor in self.preprocessors:
            tokens = preprocessor.preprocess_all(tokens, document)
        return tokens