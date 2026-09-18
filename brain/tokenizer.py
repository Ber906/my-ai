
import re
import numpy as np
from collections import defaultdict

class SimpleTokenizer:
    """Sariling tokenizer - hindi dependent sa HuggingFace"""
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = defaultdict(int)
        self.special_tokens = {
            '<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3
        }
        
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        # Tokenize and count
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_freq[word] += 1
        
        # Sort by frequency
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocab
        self.word_to_idx = {**self.special_tokens}
        self.idx_to_word = {v: k for k, v in self.special_tokens.items()}
        
        for word, _ in sorted_words[:self.vocab_size - len(self.special_tokens)]:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def _tokenize(self, text):
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def encode(self, text, max_length=50):
        """Text to numbers"""
        words = self._tokenize(text)[:max_length]
        indices = [self.word_to_idx.get(w, self.special_tokens['<UNK>']) for w in words]
        
        # Pad
        while len(indices) < max_length:
            indices.append(self.special_tokens['<PAD>'])
        
        return np.array(indices)
    
    def decode(self, indices):
        """Numbers to text"""
        words = []
        for idx in indices:
            if idx == self.special_tokens['<PAD>']:
                break
            word = self.idx_to_word.get(idx, '<UNK>')
            words.append(word)
        return ' '.join(words)
    
    def text_to_vector(self, text, embedding_dim=128):
        """Convert text to feature vector for neural network"""
        indices = self.encode(text)
        # Simple embedding: one-hot-ish
        vec = np.zeros(embedding_dim)
        for i, idx in enumerate(indices):
            if idx < embedding_dim:
                vec[idx % embedding_dim] += 1
        return vec / (np.linalg.norm(vec) + 1e-8)
