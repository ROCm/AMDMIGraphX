import numpy as np
import pickle
from pathlib import Path
import os

G_MAX_TOK_LEN = 1024
G_LLAMA2_EOS = 2

DATASET_PATH = "/dataset/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
OUTPUT_PATH = "/dataset/"

_p = Path(DATASET_PATH)
if _p.exists():
    with _p.open(mode="rb") as f:
        d = pickle.load(f)

toks = d['tok_input'].to_list()
#toks = [toks[0]]

toks_np = np.ones((len(toks), G_MAX_TOK_LEN), dtype=np.int64) * G_LLAMA2_EOS
mask_np = np.zeros((len(toks), G_MAX_TOK_LEN), dtype=np.int64)
position_nps = [np.arange(0, G_MAX_TOK_LEN, dtype=np.int64) for _ in range(len(toks))]


for i, q in enumerate(toks):
    toks_np[i, :len(q)] = q
    mask_np[i, :len(q)] = np.ones_like(q)


token_size = len(toks)

np.save(f"{OUTPUT_PATH}input_ids_size_{token_size}_seq_{G_MAX_TOK_LEN}.npy", toks_np)
np.save(f"{OUTPUT_PATH}attention_mask_size_{token_size}_seq_{G_MAX_TOK_LEN}.npy", mask_np)
np.save(f"{OUTPUT_PATH}position_ids_size_{token_size}_seq_{G_MAX_TOK_LEN}.npy", position_nps)

print("Npy filed are created")
