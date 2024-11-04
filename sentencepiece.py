import regex as re
import collections
import math
import random
from typing import List, Dict


class SentencePieceTokenizer:
    def __init__(
        self,
        vocab_size: int = 1000,
        special_tokens: List[str] = ["<s>", "</s>", "<unk>"],
        alpha: float = 0.1,
    ):  # subword regularization parameter
        """
        Initialize the tokenizer

        Args:
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens to add to vocabulary
            alpha: Smoothing parameter for subword regularization
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.alpha = alpha

        # Initialize vocabulary with special tokens
        self.vocab = {token: idx for idx, token in enumerate(special_tokens)}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

        # Initialize merges dictionary for BPE
        self.merges = {}

        # Regex for basic preprocessing
        # Handles whitespace, punctuation, and keeps numbers together
        self.pre_tokenize_pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def pre_tokenize(self, text: str) -> List[str]:
        """
        Initial tokenization of raw text into word-level tokens
        """
        return [t for t in re.findall(self.pre_tokenize_pat, text) if t.strip()]

    def compute_token_scores(self, token_freqs: Dict[str, int]) -> Dict[str, float]:
        """
        Compute language model scores for tokens using unigram probability
        """
        total_count = sum(token_freqs.values())
        scores = {}
        for token, freq in token_freqs.items():
            # Basic unigram probability with smoothing
            prob = (freq + self.alpha) / (total_count + self.alpha * len(token_freqs))
            scores[token] = math.log(prob)
        return scores

    def train(self, text: str, min_freq: int = 2):
        """
        Train the tokenizer on input text

        Args:
            text: Input text for training
            min_freq: Minimum frequency for considering a merge
        """
        # Pre-tokenize text into words using the regex
        words = self.pre_tokenize(text)

        # Initialize character-level vocabulary
        char_freqs = collections.Counter("".join(words))
        base_vocab = {
            c: i + len(self.special_tokens)
            for i, (c, _) in enumerate(char_freqs.items())
        }
        self.vocab.update(base_vocab)
        self.inv_vocab.update({i: c for c, i in base_vocab.items()})

        # Convert words to character sequences
        sequences = [[c for c in word] for word in words]

        # Track current vocabulary size
        curr_vocab_size = len(self.vocab)

        while curr_vocab_size < self.vocab_size:
            # Find most frequent pairs
            pair_freqs = collections.defaultdict(int)
            for seq in sequences:
                if len(seq) < 2:
                    continue
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_freqs[pair] += 1

            # Find best pair to merge -- the char pair that occurs the most
            if not pair_freqs:
                break

            best_pair = max(pair_freqs.items(), key=lambda x: x[1])
            if best_pair[1] < min_freq:
                break

            # Create new token and add to vocabulary
            new_token = "".join(best_pair[0])
            self.vocab[new_token] = curr_vocab_size
            self.inv_vocab[curr_vocab_size] = new_token
            self.merges[best_pair[0]] = curr_vocab_size

            # Update sequences with merged pairs
            new_sequences = []
            for seq in sequences:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair[0]:
                        new_seq.append(new_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences
            curr_vocab_size += 1

    def encode(self, text: str, sample: bool = False) -> List[int]:
        """
        Encode text to token ids

        Args:
            text: Text to encode
            sample: Whether to use subword regularization
        Returns:
            List of token ids
        """
        if not text:
            return []

        # Pre-tokenize
        words = self.pre_tokenize(text)

        # Initialize with character-level tokens
        sequences = [[c for c in word] for word in words]

        # Apply merges
        for seq in sequences:
            i = 0
            while i < len(seq) - 1:
                current_pair = (seq[i], seq[i + 1])
                if current_pair in self.merges:
                    # If sampling enabled, probabilistically skip some merges
                    if sample and random.random() < self.alpha:
                        i += 1
                        continue

                    new_token = "".join(current_pair)
                    seq[i : i + 2] = [new_token]
                else:
                    i += 1

        # Flatten and convert to ids
        tokens = []
        for seq in sequences:
            for token in seq:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab["<unk>"])

        return tokens

    def decode(self, ids: List[int]) -> str:
        """
        Decode token ids back to text

        Args:
            ids: List of token ids
        Returns:
            Decoded text
        """
        tokens = []
        for idx in ids:
            if idx in self.inv_vocab:
                tokens.append(self.inv_vocab[idx])
            else:
                tokens.append(self.inv_vocab[self.vocab["<unk>"]])
        return "".join(tokens)


# Usage Example
if __name__ == "__main__":
    # Sample text for testing
    # text = """SentencePiece is an unsupervised text tokenizer and detokenizer.
    # It implements subword units like BPE and unigram language model with the
    # extension of direct training from raw sentences."""

    # read the tokenizer dataset
    with open("input_text.txt", "r") as ip_file:
        text = ip_file.readlines()
    text = text[:-1]

    # Initialize and train tokenizer
    tokenizer = SentencePieceTokenizer(vocab_size=100)
    tokenizer.train(text)

    # Test encoding and decoding
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Original text: {text}")
    print(f"Encoded: {encoded[:10]}...")
    print(f"Decoded text: {decoded}")

    # Test with subword regularization
    encoded_sampled = tokenizer.encode(text, sample=True)
    print(f"Encoded (with sampling): {encoded_sampled[:10]}...")
