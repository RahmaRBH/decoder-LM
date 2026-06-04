# prepare_data.py
# Downloads the Python split of CodeSearchNet, tokenizes it with tiktoken,
# and saves train.bin + val.bin as numpy uint16 arrays for memory-mapped loading.
# Run once before training: python3 prepare_data.py

import os
import numpy as np
import tiktoken
from datasets import load_dataset

DATA_DIR = "."   # train.bin and val.bin are written here
SPLIT    = 0.9  # 90% train, 10% val

print("Loading CodeSearchNet (python)...")
dataset = load_dataset("code_search_net", "python")

functions = []
for split in ("train", "validation", "test"):
    for example in dataset[split]:
        func = example["whole_func_string"].strip()
        if func:
            functions.append(func)

print(f"Total functions: {len(functions):,}")

corpus = "\n\n".join(functions)

print("Tokenizing with tiktoken (gpt2)...")
enc    = tiktoken.get_encoding("gpt2")
tokens = enc.encode_ordinary(corpus)  # encode_ordinary skips special tokens
tokens = np.array(tokens, dtype=np.uint16)  # uint16 fits gpt2 vocab (50257 < 65535)

print(f"Total tokens: {len(tokens):,}")

n           = int(SPLIT * len(tokens))
train_ids   = tokens[:n]
val_ids     = tokens[n:]

train_path  = os.path.join(DATA_DIR, "train.bin")
val_path    = os.path.join(DATA_DIR, "val.bin")

train_ids.tofile(train_path)
val_ids.tofile(val_path)

print(f"train.bin: {os.path.getsize(train_path) / 1e6:.1f} MB  ({len(train_ids):,} tokens)")
print(f"val.bin:   {os.path.getsize(val_path)   / 1e6:.1f} MB  ({len(val_ids):,} tokens)")
print("Done — run: python3 v2.py")
