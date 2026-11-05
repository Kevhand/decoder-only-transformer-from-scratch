"""
Byte-level BPE tokenizer with a Unicode-aware pre-tokenizer.

Usage:
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=1000)
    tokenizer.train(corpus_text)
    tokenizer.save("tokenizer.json")

    tokenizer = ByteLevelBPETokenizer.load("tokenizer.json")
    ids = tokenizer.encode("Hello world!")          # raw ids, no special tokens
    seq = tokenizer.prepare_sequence(ids, 128)      # [BOS] ids [EOS] [PAD]...
    text = tokenizer.decode(seq)
"""

import heapq
import json
import unicodedata
from collections import Counter


# ============================================================
# Pre-tokenizer
# ============================================================

KNOWN_CONTRACTIONS = ["'s", "'re", "'ve", "'ll", "'d", "n't", "'m"]

# Bump this whenever pre_tokenize() changes behaviour. Saved tokenizers record
# it, and loading a tokenizer trained with a different version is refused,
# because the same text would split into different chunks.
PRE_TOKENIZER_VERSION = 1


def check_contractions(text, i):
    for contraction in KNOWN_CONTRACTIONS:
        if text.startswith(contraction, i):
            if i == 0 or not text[i - 1].isalnum():
                continue
            if i + len(contraction) == len(text) or not text[i + len(contraction)].isalnum():
                return contraction
    return None


def is_letter(ch):
    return unicodedata.category(ch).startswith("L")


def is_number(ch):
    return unicodedata.category(ch).startswith("N")


def is_symbol(ch):
    return not ch.isspace() and not is_letter(ch) and not is_number(ch)


def pre_tokenize(text):
    tokens = []
    i = 0

    while i < len(text):
        if text[i] == "\n":
            tokens.append(text[i])
            i += 1
            continue

        if is_symbol(text[i]):
            contraction = check_contractions(text, i)
            if contraction:
                tokens.append(contraction)
                i += len(contraction)
                continue

            symbols = ""
            while i < len(text) and is_symbol(text[i]):
                symbols += text[i]
                i += 1
            tokens.append(symbols)
            continue

        if is_number(text[i]):
            number = ""
            while i < len(text) and is_number(text[i]):
                number += text[i]
                i += 1
            tokens.append(number)
            continue

        if text[i].isspace():
            whitespace = text[i]
            i += 1

            if i < len(text) and not text[i].isspace() and is_letter(text[i]):
                word = whitespace
                while i < len(text) and not text[i].isspace() and is_letter(text[i]):
                    contraction = check_contractions(text, i)
                    if contraction:
                        break
                    word += text[i]
                    i += 1
                tokens.append(word)

                contraction = check_contractions(text, i)
                if contraction:
                    tokens.append(contraction)
                    i += len(contraction)
                    continue
            else:
                contraction = check_contractions(text, i)
                if contraction:
                    tokens.append(whitespace)
                    continue
                if i < len(text) and is_symbol(text[i]):
                    symbols = whitespace
                    while i < len(text) and is_symbol(text[i]):
                        symbols += text[i]
                        i += 1
                    tokens.append(symbols)
                elif i < len(text) and is_number(text[i]):
                    number = whitespace
                    while i < len(text) and is_number(text[i]):
                        number += text[i]
                        i += 1
                    tokens.append(number)
                else:
                    tokens.append(whitespace)
            continue

        word = ""
        while i < len(text) and not text[i].isspace() and is_letter(text[i]):
            contraction = check_contractions(text, i)
            if contraction:
                break
            word += text[i]
            i += 1
        tokens.append(word)

        contraction = check_contractions(text, i)
        if contraction:
            tokens.append(contraction)
            i += len(contraction)
            continue

    return tokens


def text_to_bytes(text):
    data = text.encode("utf-8")
    return [data[i:i + 1] for i in range(len(data))]


def text_to_byte_sequences(text):
    # Empty chunks can appear in edge cases like "123n't"; they carry no bytes.
    return [text_to_bytes(chunk) for chunk in pre_tokenize(text) if chunk]


def count_chunks(texts):
    """Count how often each pre-token chunk occurs across an iterable of texts.

    Pre-tokenize each document separately, so chunks never span documents.
    The resulting Counter is all BPE training needs, and Counters from
    different worker processes can simply be added together.
    """
    if isinstance(texts, str):
        texts = [texts]

    counts = Counter()
    for text in texts:
        counts.update(chunk for chunk in pre_tokenize(text) if chunk)
    return counts


# ============================================================
# Pair-counting helpers used during training
# ============================================================

def count_sequence_pairs(sequence):
    count = Counter()
    for i in range(len(sequence) - 1):
        count[(sequence[i], sequence[i + 1])] += 1
    return count


def merge_single_sequence(sequence, pair):
    merged = []
    i = 0
    while i < len(sequence):
        if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair:
            merged.append(sequence[i] + sequence[i + 1])
            i += 2
        else:
            merged.append(sequence[i])
            i += 1
    return merged


# ============================================================
# Tokenizer
# ============================================================

class ByteLevelBPETokenizer:
    DEFAULT_SPECIAL_TOKENS = [b"[PAD]", b"[BOS]", b"[EOS]", b"[UNK]"]
    FORMAT_VERSION = 1

    # Encoding caches chunk -> ids. Word frequencies are heavily skewed, so most
    # chunks repeat and BPE runs once per distinct chunk. The cache is cleared
    # when it reaches this many entries, which bounds memory.
    CACHE_SIZE = 1_000_000

    def __init__(self, target_vocab_size=None, special_tokens=None):
        # The size requested for training. The actual size can be smaller if
        # the corpus runs out of pairs to merge; see the vocab_size property.
        self.target_vocab_size = target_vocab_size
        self.special_tokens = list(special_tokens or self.DEFAULT_SPECIAL_TOKENS)

        # The only learned state: ordered merges as (left_id, right_id).
        self.merges = []

        self._build_lookups()

    # --------------------------------------------------------
    # Vocabulary construction
    # --------------------------------------------------------

    def _base_vocab(self):
        # ids 0-255: raw bytes, then special tokens in order (PAD = 256, ...)
        return [bytes([i]) for i in range(256)] + list(self.special_tokens)

    def _build_lookups(self):
        """Rebuild vocab, stoi, itos and merge_rank from self.merges."""
        vocab = self._base_vocab()
        merge_rank = {}

        for rank, (left_id, right_id) in enumerate(self.merges):
            left, right = vocab[left_id], vocab[right_id]
            vocab.append(left + right)
            merge_rank[(left, right)] = rank

        self.vocab = vocab
        self.itos = {i: token for i, token in enumerate(vocab)}

        # If two merge paths ever produce the same bytes, keep the first id
        # so encoding is stable. Decoding works for either id.
        self.stoi = {}
        for i, token in enumerate(vocab):
            self.stoi.setdefault(token, i)

        self.merge_rank = merge_rank
        self.special_ids = {self.stoi[token] for token in self.special_tokens}

        # Merges changed, so any cached encodings are stale.
        self._cache = {}

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def train(self, text, verbose=False):
        """Train on a single string or an iterable of documents."""
        return self.train_from_counts(count_chunks(text), verbose=verbose)

    def train_from_counts(self, chunk_counts, verbose=False):
        """Train from a Counter of pre-token chunk -> number of occurrences.

        Each distinct chunk is stored once with a weight, so the work depends on
        the number of distinct chunks, not on the size of the corpus. The merges
        are identical to training on every occurrence separately.
        """
        base_size = len(self._base_vocab())

        if self.target_vocab_size is None or self.target_vocab_size < base_size:
            raise ValueError(f"target_vocab_size must be at least {base_size} (256 bytes + special tokens).")

        num_merges = self.target_vocab_size - base_size

        self.merges = []
        self._build_lookups()

        # Local id table that grows as merges are learned.
        token_to_id = dict(self.stoi)
        next_id = base_size

        # Sort so the result never depends on Counter insertion order.
        chunks = sorted(chunk_counts)
        byte_sequences = [text_to_bytes(chunk) for chunk in chunks]
        weights = [chunk_counts[chunk] for chunk in chunks]

        # Initial pair counts and the index of sequences containing each pair.
        pair_counts = Counter()
        pair_locations = {}

        for index, sequence in enumerate(byte_sequences):
            weight = weights[index]
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                pair_counts[pair] += weight
                pair_locations.setdefault(pair, set()).add(index)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        while len(self.merges) < num_merges:
            best_pair = self._pop_best_pair(heap, pair_counts)
            if best_pair is None:
                break

            changed_pairs = self._merge_pair(best_pair, byte_sequences, weights, pair_counts, pair_locations)

            for pair in changed_pairs:
                if pair in pair_counts:
                    heapq.heappush(heap, (-pair_counts[pair], pair))

            left_id = token_to_id[best_pair[0]]
            right_id = token_to_id[best_pair[1]]
            self.merges.append((left_id, right_id))

            merged_token = best_pair[0] + best_pair[1]
            token_to_id.setdefault(merged_token, next_id)
            next_id += 1

            if verbose and len(self.merges) % verbose == 0:
                print(f"merge {len(self.merges)}/{num_merges}: {merged_token!r}")

        self._build_lookups()

        if self.vocab_size < self.target_vocab_size:
            print(
                f"Warning: stopped at {self.vocab_size} of {self.target_vocab_size} "
                f"requested tokens; no pairs left to merge."
            )

        return self

    @staticmethod
    def _pop_best_pair(heap, pair_counts):
        # Lazy deletion: skip heap entries whose count is out of date.
        while heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) == -neg_count:
                return pair
        return None

    @staticmethod
    def _merge_pair(pair, byte_sequences, weights, pair_counts, pair_locations):
        changed_pairs = set()
        affected = pair_locations.get(pair, set()).copy()

        for index in affected:
            sequence = byte_sequences[index]
            weight = weights[index]

            # Remove this sequence's old pairs from the counts and index.
            for old_pair, count in count_sequence_pairs(sequence).items():
                changed_pairs.add(old_pair)
                pair_counts[old_pair] -= count * weight
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]

                pair_locations[old_pair].discard(index)
                if not pair_locations[old_pair]:
                    del pair_locations[old_pair]

            new_sequence = merge_single_sequence(sequence, pair)
            byte_sequences[index] = new_sequence

            # Add the merged sequence's pairs back.
            for new_pair, count in count_sequence_pairs(new_sequence).items():
                changed_pairs.add(new_pair)
                pair_counts[new_pair] += count * weight
                pair_locations.setdefault(new_pair, set()).add(index)

        return changed_pairs

    # --------------------------------------------------------
    # Encoding and decoding
    # --------------------------------------------------------

    def _apply_merges(self, sequence):
        while len(sequence) > 1:
            best_pair = None
            best_rank = float("inf")

            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            sequence = merge_single_sequence(sequence, best_pair)

        return sequence

    def encode(self, text):
        """Text -> raw token ids. No BOS, EOS or padding."""
        ids = []
        cache = self._cache
        unk_id = self.unk_id

        for chunk in pre_tokenize(text):
            if not chunk:
                continue

            chunk_ids = cache.get(chunk)
            if chunk_ids is None:
                pieces = self._apply_merges(text_to_bytes(chunk))
                chunk_ids = [self.stoi.get(piece, unk_id) for piece in pieces]

                if len(cache) >= self.CACHE_SIZE:
                    cache.clear()
                cache[chunk] = chunk_ids

            ids.extend(chunk_ids)

        return ids

    def prepare_sequence(self, token_ids, max_length):
        """[BOS] + ids + [EOS], truncated and right-padded to exactly max_length."""
        if max_length < 2:
            raise ValueError("max_length must be at least 2 to fit [BOS] and [EOS].")

        content_length = max_length - 2
        token_ids = list(token_ids[:content_length])
        padding_needed = content_length - len(token_ids)

        return [self.bos_id] + token_ids + [self.eos_id] + [self.pad_id] * padding_needed

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []

        for token_id in token_ids:
            if token_id not in self.itos:
                raise ValueError(f"Unknown token id: {token_id}")
            if skip_special_tokens and token_id in self.special_ids:
                continue
            pieces.append(self.itos[token_id])

        return b"".join(pieces).decode("utf-8", errors="replace")

    def __call__(self, text):
        return self.encode(text)

    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        """Actual number of vocabulary entries. Size the model from this."""
        return len(self.vocab)

    # --------------------------------------------------------
    # Special token ids
    # --------------------------------------------------------

    @property
    def pad_id(self):
        return self.stoi[b"[PAD]"]

    @property
    def bos_id(self):
        return self.stoi[b"[BOS]"]

    @property
    def eos_id(self):
        return self.stoi[b"[EOS]"]

    @property
    def unk_id(self):
        return self.stoi[b"[UNK]"]

    # --------------------------------------------------------
    # Save and load
    # --------------------------------------------------------

    def save(self, path):
        data = {
            "type": "byte_level_bpe",
            "format_version": self.FORMAT_VERSION,
            "pre_tokenizer_version": PRE_TOKENIZER_VERSION,
            "contractions": KNOWN_CONTRACTIONS,
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "special_tokens": [token.decode("utf-8") for token in self.special_tokens],
            "merges": [[left, right] for left, right in self.merges],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("type") != "byte_level_bpe":
            raise ValueError("Not a byte-level BPE tokenizer file.")
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(f"Unsupported format version: {data.get('format_version')}")
        if data.get("pre_tokenizer_version") != PRE_TOKENIZER_VERSION:
            raise ValueError(
                "Tokenizer was trained with a different pre-tokenizer version; "
                "the same text would split differently."
            )
        if data.get("contractions") != KNOWN_CONTRACTIONS:
            raise ValueError("Tokenizer was trained with a different contraction list.")

        special_tokens = [token.encode("utf-8") for token in data["special_tokens"]]
        # Files from before the target/actual split stored the target as "vocab_size".
        target = data.get("target_vocab_size", data.get("vocab_size"))
        tokenizer = cls(target_vocab_size=target, special_tokens=special_tokens)
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer._build_lookups()

        return tokenizer
