"""
AutoLab Data Preparation Pipeline
=================================

This script prepares the dataset and tokenizer used by AutoLab
autonomous research experiments.

The pipeline performs two steps:

1. Download dataset shards
2. Train a BPE tokenizer

All artifacts are stored in:

    ~/.cache/autolab/

Usage
-----

Full preparation:

    python prepare.py

Download only small dataset for testing:

    python prepare.py --num-shards 8

Download full dataset:

    python prepare.py --num-shards -1


AutoLab Design Philosophy
-------------------------

This script is intentionally designed to be **simple and agent-editable**.

AI research agents may:

• change tokenizer vocabulary size
• change tokenizer pattern
• change dataset sampling
• introduce new preprocessing steps

These modifications allow AutoLab agents to experiment
with different research ideas autonomously.

"""

import os
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch


# =============================================================================
# AutoLab Global Configuration
# =============================================================================

# Sequence length used across experiments
MAX_SEQ_LEN = 2048

# training runtime budget (seconds)
TIME_BUDGET = 300

# evaluation tokens
EVAL_TOKENS = 40 * 524288

# =============================================================================
# Cache Directories
# =============================================================================

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autolab")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# dataset source
BASE_URL = (
    "https://huggingface.co/datasets/"
    "karpathy/climbmix-400b-shuffle/resolve/main"
)

MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD

VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"

VOCAB_SIZE = 8192

# =============================================================================
# Tokenizer configuration
# =============================================================================

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]

BOS_TOKEN = "<|reserved_0|>"

# =============================================================================
# Utility Logging
# =============================================================================


def log(msg):
    print(f"[AutoLab] {msg}")


# =============================================================================
# Dataset Download
# =============================================================================


def download_single_shard(index):
    """
    Download one dataset shard.

    The function retries several times in case of network errors.
    """

    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            temp_path = filepath + ".tmp"

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

            os.rename(temp_path, filepath)

            log(f"Downloaded {filename}")

            return True

        except (requests.RequestException, IOError) as e:

            log(f"Download failed ({attempt}/{max_attempts}) for {filename}: {e}")

            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            if attempt < max_attempts:
                time.sleep(2 ** attempt)

    return False


def download_data(num_shards, download_workers=8):
    """
    Download training shards and pinned validation shard.
    """

    os.makedirs(DATA_DIR, exist_ok=True)

    num_train = min(num_shards, MAX_SHARD)

    ids = list(range(num_train))

    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    existing = sum(
        1
        for i in ids
        if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet"))
    )

    if existing == len(ids):

        log(f"All {len(ids)} shards already exist.")

        return

    needed = len(ids) - existing

    log(f"Downloading {needed} shards ({existing} already present)...")

    workers = max(1, min(download_workers, needed))

    with Pool(processes=workers) as pool:

        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)

    log(f"{ok}/{len(ids)} shards ready.")


# =============================================================================
# Dataset Iterator
# =============================================================================


def list_parquet_files():
    files = sorted(
        f
        for f in os.listdir(DATA_DIR)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )

    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """
    Stream documents for tokenizer training.
    """

    parquet_paths = [
        p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)
    ]

    nchars = 0

    for filepath in parquet_paths:

        pf = pq.ParquetFile(filepath)

        for rg_idx in range(pf.num_row_groups):

            rg = pf.read_row_group(rg_idx)

            for text in rg.column("text").to_pylist():

                doc = text[:doc_cap] if len(text) > doc_cap else text

                nchars += len(doc)

                yield doc

                if nchars >= max_chars:
                    return


# =============================================================================
# Tokenizer Training
# =============================================================================


def train_tokenizer():

    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")

    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):

        log("Tokenizer already trained.")

        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()

    if len(parquet_files) < 2:

        log("Need at least two shards (train + val).")

        sys.exit(1)

    log("Training BPE tokenizer...")

    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()

    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)

    tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size_no_special,
        pattern=SPLIT_PATTERN,
    )

    pattern = tokenizer.get_pattern()

    mergeable_ranks = {
        bytes(k): v for k, v in tokenizer.get_mergeable_ranks()
    }

    tokens_offset = len(mergeable_ranks)

    special_tokens = {
        name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
    }

    enc = tiktoken.Encoding(
        name="autolab_bpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    log(f"Tokenizer trained in {time.time()-t0:.1f}s")

    log("Building token_bytes lookup...")

    special_set = set(SPECIAL_TOKENS)

    token_bytes_list = []

    for token_id in range(enc.n_vocab):

        token_str = enc.decode([token_id])

        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))

    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)

    torch.save(token_bytes_tensor, token_bytes_path)

    log("Tokenizer artifacts saved.")

    # sanity check

    test = "Hello world! 你好 123"

    encoded = enc.encode_ordinary(test)

    decoded = enc.decode(encoded)

    assert decoded == test

    log("Tokenizer sanity check passed.")


# =============================================================================
# Tokenizer Wrapper (Runtime)
# =============================================================================


class Tokenizer:

    def __init__(self, enc):

        self.enc = enc

        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):

        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:

            enc = pickle.load(f)

        return cls(enc)

    def get_vocab_size(self):

        return self.enc.n_vocab

    def get_bos_token_id(self):

        return self.bos_token_id

    def encode(self, text, prepend=None):

        if isinstance(text, str):

            ids = self.enc.encode_ordinary(text)

            if prepend is not None:

                ids.insert(0, prepend)

            return ids

        if isinstance(text, list):

            return self.enc.encode_ordinary_batch(text)

        raise ValueError("Unsupported input type")

    def decode(self, ids):

        return self.enc.decode(ids)


# =============================================================================
# Evaluation Metric
# =============================================================================


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):

    token_bytes = torch.load(
        os.path.join(TOKENIZER_DIR, "token_bytes.pt"),
        map_location="cuda",
    )

    from train import make_dataloader

    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")

    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)

    total_nats = 0.0
    total_bytes = 0

    for _ in range(steps):

        x, y, _ = next(val_loader)

        loss_flat = model(x, y, reduction="none").view(-1)

        y_flat = y.view(-1)

        nbytes = token_bytes[y_flat]

        mask = nbytes > 0

        total_nats += (loss_flat * mask).sum().item()

        total_bytes += nbytes.sum().item()

    return total_nats / (math.log(2) * total_bytes)


# =============================================================================
# Main
# =============================================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="number of shards to download (-1 = all)",
    )

    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    log(f"Cache directory: {CACHE_DIR}")

    download_data(num_shards, args.download_workers)

    train_tokenizer()

    log("AutoLab preparation finished.")


if __name__ == "__main__":

    main()
  
