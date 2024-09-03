import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
# Variables and constants
SHARD_SIZE = int(1e8)  # 100M tokens per shard, total of 100 shards
NUM_PROCS = max(1, os.cpu_count() // 2)  # Number of processes for multiprocessing

DATA_CACHE_DIR = "./wikitext-small"
DATASET_CACHE_DIR = "./datasets/"
DATASET_NAME = "Salesforce/wikitext"
REMOTE_NAME = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC._special_tokens["<|endoftext|>"]  # end of text token.

os.makedirs(DATA_CACHE_DIR, exist_ok=True)


def tokenize(doc):
    tokens = [EOT]  # Add the special <|endoftext|> token
    tokens.extend(ENC.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def process_shards(dataset, shard_size, num_procs):
    shard_index = 0
    token_count = 0

    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)

    total_tokens_to_process = len(dataset) * shard_size
    overall_progress_bar = tqdm(
        total=total_tokens_to_process, unit="tokens", desc="Overall Progress"
    )
    with mp.Pool(num_procs) as pool:
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            overall_progress_bar.update(len(tokens))

            if token_count > 100000:
                break

            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)

            else:
                remainder = shard_size - token_count
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(create_filename(shard_index), all_tokens_np)
                shard_index += 1
                all_tokens_np[: len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count > 0:
            write_datafile(create_filename(shard_index), all_tokens_np[:token_count])
    overall_progress_bar.close()


def create_filename(shard_index):
    """Generates the filename for a shard."""
    split = "val" if shard_index == 0 else "train"
    return os.path.join(DATA_CACHE_DIR, f"wikitext_{split}_{shard_index:01d}")


if __name__ == "__main__":
    fw = load_dataset(
        DATASET_NAME, cache_dir=DATASET_CACHE_DIR, name=REMOTE_NAME, split=DATASET_SPLIT
    )
    process_shards(fw, SHARD_SIZE, NUM_PROCS)
