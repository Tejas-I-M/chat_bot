# =============================================================================
# src/utils.py — Data Pipeline, Character-Level Tokenizer & Batch Generator
# =============================================================================
#
# This module handles three core responsibilities:
#   1. Auto-downloading the TinyShakespeare corpus if not present locally
#   2. Building and persisting a character-level vocabulary (stoi / itos maps)
#   3. Providing an efficient random batch sampler for language model training
#
# Design Note:
#   Character-level tokenization keeps the vocabulary small (≈65 chars) which is
#   ideal for learning the fundamentals of autoregressive language modeling without
#   the complexity of BPE or SentencePiece tokenizers used in production LLMs.
# =============================================================================

import os
import pickle
import requests
import numpy as np
import torch
from typing import Tuple, Dict, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Official TinyShakespeare dataset — ~1MB of Shakespeare's collected works,
# widely used as a benchmark for character-level language models.
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "input.txt")
META_FILE = os.path.join(DATA_DIR, "meta.pkl")


# ---------------------------------------------------------------------------
# 1. Dataset Downloader
# ---------------------------------------------------------------------------

def ensure_data_exists() -> str:
    """
    Checks whether 'data/input.txt' exists on disk.
    If not found, downloads TinyShakespeare from the official GitHub mirror.

    Returns:
        str: Raw text content of the dataset.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"[utils] Dataset not found locally. Downloading from:\n  {DATASET_URL}")
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()                         # Raise on HTTP errors
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"[utils] Saved {len(response.text):,} characters to '{DATA_FILE}'")
    else:
        print(f"[utils] Dataset found at '{DATA_FILE}'. Skipping download.")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 2. Character-Level Tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(text: str) -> Tuple[Dict, Dict, int]:
    """
    Constructs a character-level vocabulary from the full corpus text.

    The vocabulary is simply all unique characters sorted lexicographically.
    Each character gets a unique integer ID:
      - stoi (str → int): encoding map used during training
      - itos (int → str): decoding map used during generation

    This is the same approach used by Karpathy's nanoGPT and char-rnn.

    Args:
        text (str): Full raw text corpus.

    Returns:
        stoi (dict): Character-to-index mapping.
        itos (dict): Index-to-character mapping.
        vocab_size (int): Number of unique characters in the corpus.
    """
    # Collect the sorted set of all unique characters in the corpus
    chars      = sorted(set(text))
    vocab_size = len(chars)

    # Build forward (encoding) and reverse (decoding) lookup tables
    stoi = {ch: i for i, ch in enumerate(chars)}   # e.g. {'A': 0, 'B': 1, ...}
    itos = {i: ch for i, ch in enumerate(chars)}   # e.g. {0: 'A', 1: 'B', ...}

    print(f"[utils] Vocabulary size: {vocab_size} unique characters")
    print(f"[utils] Sample chars: {chars[:10]} ...")

    return stoi, itos, vocab_size


def encode(text: str, stoi: Dict) -> List[int]:
    """Converts a string into a list of integer token IDs."""
    return [stoi[c] for c in text]


def decode(indices: List[int], itos: Dict) -> str:
    """Converts a list of integer token IDs back into a string."""
    return "".join([itos[i] for i in indices])


def save_tokenizer(stoi: dict, itos: dict, vocab_size: int) -> None:
    """
    Persists the tokenizer vocabulary to 'data/meta.pkl'.

    Serializing these mappings avoids rebuilding the vocabulary each time
    the app or training script is launched — important for reproducibility.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    meta = {"stoi": stoi, "itos": itos, "vocab_size": vocab_size}
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
    print(f"[utils] Tokenizer saved to '{META_FILE}'")


def load_tokenizer() -> Tuple[Dict, Dict, int]:
    """
    Loads the pre-built tokenizer from 'data/meta.pkl'.

    Raises:
        FileNotFoundError: If meta.pkl does not exist (run train_script.py first).
    """
    if not os.path.exists(META_FILE):
        raise FileNotFoundError(
            f"Tokenizer not found at '{META_FILE}'. "
            "Please run 'train_script.py' first to build and save it."
        )
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return meta["stoi"], meta["itos"], meta["vocab_size"]


# ---------------------------------------------------------------------------
# 3. Batch Generator
# ---------------------------------------------------------------------------

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly samples a mini-batch of (input, target) sequence pairs.

    Why random sampling?
        For language model pretraining we treat the corpus as one long sequence.
        Random offsets ensure diverse context windows across batches, which
        stabilizes gradient estimates and reduces overfitting.

    Offset-by-one target construction:
        Given an input sequence [t₀, t₁, ..., t_{L-1}], the target is
        [t₁, t₂, ..., t_L]. This forces the model to predict the *next*
        token at every position — the core autoregressive objective.

    Args:
        data       (Tensor): Full encoded corpus as a 1-D int64 tensor.
        block_size (int):    Context window length (max tokens per sequence).
        batch_size (int):    Number of sequences per mini-batch.
        device     (device): Target device ('cuda' or 'cpu').

    Returns:
        x (Tensor): Input sequences  of shape (batch_size, block_size).
        y (Tensor): Target sequences of shape (batch_size, block_size).
    """
    # Sample random starting positions, ensuring there's room for block_size+1 tokens
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stack individual sequences into a 2-D batch tensor
    x = torch.stack([data[i    : i + block_size    ] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

    # Move tensors to the appropriate device (GPU or CPU)
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 4. Train / Validation Split Helper
# ---------------------------------------------------------------------------

def prepare_datasets(
    text: str,
    stoi: Dict,
    train_frac: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes the full text and splits it into train / validation tensors.

    A 90/10 split is standard for small corpora like TinyShakespeare.
    The validation set is used to detect overfitting during training.

    Args:
        text       (str):   Raw corpus text.
        stoi       (dict):  Character-to-index mapping.
        train_frac (float): Fraction of data used for training.

    Returns:
        train_data (Tensor): Encoded training corpus.
        val_data   (Tensor): Encoded validation corpus.
    """
    encoded = encode(text, stoi)
    data    = torch.tensor(encoded, dtype=torch.long)

    n          = int(train_frac * len(data))
    train_data = data[:n]
    val_data   = data[n:]

    print(f"[utils] Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
    return train_data, val_data
