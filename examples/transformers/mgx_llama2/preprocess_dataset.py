#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
import numpy as np
import pickle
from pathlib import Path

G_MAX_TOK_LEN = 1024
G_LLAMA2_EOS = 2

DATASET_PATH = "/dataset/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
OUTPUT_PATH = "/dataset/"

_p = Path(DATASET_PATH)
if _p.exists():
    with _p.open(mode="rb") as f:
        d = pickle.load(f)
else:
    raise RuntimeError(f"Missing dataset from {DATASET_PATH}")

toks = d['tok_input'].to_list()

toks_np = np.ones((len(toks), G_MAX_TOK_LEN), dtype=np.int64) * G_LLAMA2_EOS
mask_np = np.zeros((len(toks), G_MAX_TOK_LEN), dtype=np.int64)
position_nps = [
    np.arange(0, G_MAX_TOK_LEN, dtype=np.int64) for _ in range(len(toks))
]

for i, q in enumerate(toks):
    toks_np[i, :len(q)] = q
    mask_np[i, :len(q)] = np.ones_like(q)

token_size = len(toks)

np.save(f"{OUTPUT_PATH}input_ids_size_{token_size}_seq_{G_MAX_TOK_LEN}.npy",
        toks_np)
np.save(
    f"{OUTPUT_PATH}attention_mask_size_{token_size}_seq_{G_MAX_TOK_LEN}.npy",
    mask_np)
np.save(f"{OUTPUT_PATH}position_ids_size_{token_size}_seq_{G_MAX_TOK_LEN}.npy",
        position_nps)

print("Npy files are created")
