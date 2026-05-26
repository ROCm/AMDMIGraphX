# CK vs MLIR split-KV benchmark analysis (2, 4, and 8 splits)

This note summarizes patterns from the generated stats files `stats_2_splits.csv`, `stats_4_splits.csv`, and `stats_8_splits.csv` in this directory, produced by `benchmarks/scripts/analyze_ck_mlir_benchmark_csvs.py` on the corresponding `benchmark_splitkv_ck_vs_mlir_*_splits.csv` inputs.

**Definitions:** “Definitively faster” means outside the script’s default **±5%** neutral band on `speedup_pct` (i.e. `ck_win` vs `mlir_win` in the stats). Rows inside ±5% are treated as **neutral** / equivalent for this analysis.

---

## 2 splits (576 rows)

**Verdict:** 35.8% neutral, 25.2% CK, 39.1% MLIR. Among **non-neutral** rows, MLIR wins about **61%** of the time (225 vs 145).

### Where CK tends to be clearly faster

- **`O` small:** e.g. **O=32** → 48 CK vs 2 MLIR material wins; **O=48** and **O=64** also CK-heavy.
- **Geometry `O < K`:** buckets **`2<K/O<=4`** and **`1<K/O<=2`** show many more CK than MLIR material wins (e.g. 49 vs 5 and 47 vs 26 in the coarse O/K table).
- **`N` large helps CK somewhat:** **N=4096** has the strongest CK share of the three **N** slices (72 vs 78 MLIR), still slightly MLIR overall.

### Where MLIR tends to be clearly faster

- **`O` large:** **O=192** (0 vs 70) and **O=256** (1 vs 69) among material wins.
- **`O > K`:** **`2<O/K<=4`** (0 vs 74), **`O/K>4`** (0 vs 27); **`1<O/K<=2`** is also MLIR-heavy (9 vs 66).
- **`N=1024`:** very MLIR-skewed among material wins (11 vs 65).

### (K, O) grid

Same story at cell level: small **O** skews CK; for **O** of 192 or 256, cells are almost all MLIR material wins.

---

## 4 splits (576 rows)

**Verdict:** **Neutrals rise to 52.3%**; 16.5% CK and 31.2% MLIR overall. Among **non-neutral**, MLIR wins about **65.5%** vs CK **34.5%**.

### CK (still visible)

- **Small `O`:** **O=32** remains **28 vs 6** (CK); **O=48** / **O=64** stay CK-favorable in counts.
- **`N=4096`:** **66 vs 75** — CK is closest to MLIR here (same direction as 2 splits).

### MLIR

- **Large `O`:** **O=256** **2 vs 49**; **O=192** still MLIR-heavy (**3 vs 46**).
- **`2<O/K<=4` / `O/K>4`:** again almost all MLIR on the material side (**1 vs 51**, **2 vs 19**); **`1<O/K<=2`** **13 vs 44**.

### Note on noise

With about **half** of rows neutral, definitive conclusions apply to **fewer** cases than at 2 splits, but **O** and **O/K** structure **match** the 2-split story.

---

## 8 splits (576 rows)

**Verdict:** **~80% neutral**; only **4.9%** CK and **15.5%** MLIR overall. Among the **117** non-neutral rows, MLIR wins about **76%** (89 vs 28).

### CK (rare)

- One of the few slices where **CK > MLIR** in material-win counts is **O=32** (**9 vs 6**).
- **`K/O>4`** bucket shows **8 CK, 0 MLIR** material wins (a narrow corner of the grid with few decisive rows).
- **`N=4096`** again has the **most** CK material wins (**22 vs 59**) vs **N=1024** / **N=2048** (4 vs 16, 2 vs 14).

### MLIR

- **`O=256`:** **1 vs 28**; **`O=192`:** **1 vs 24**.
- **`K=32`:** **0 vs 12** — at 8 splits, that slice is almost entirely neutral or MLIR when not.
- **`2<O/K<=4`**, **`O/K>4`**, and **`O=K`** material rows skew heavily **MLIR** (e.g. **`O=K`** **0 vs 14**).

### Interpretation

With **8 splits**, CK and MLIR timings are **much closer** most of the time, so “definitively faster” is **uncommon**. When the benchmark **does** pick a side outside ±5%, it is **still usually MLIR**.

---

## Overarching patterns (2 → 4 → 8 splits)

1. **Head / output width (`O`) is the main axis**  
   **Small `O`** → CK wins a **large share** of **material** comparisons. **Large `O`** → MLIR dominates. That holds at **2** and **4** splits; at **8** splits most rows are neutral, but **O=32** vs **O=256** still shows the same **direction** in the few decisive rows.

2. **`O` vs `K` ratio**  
   **`O > K`** bands (**`1<O/K<=2`**, **`2<O/K<=4`**, **`O/K>4`**) are **consistently MLIR-heavy** when there is a winner. **`O < K`** bands are where **CK** is **competitive or ahead** (especially **`2<K/O<=4`** and **`1<K/O<=2`** at 2 and 4 splits).

3. **Sequence length `N`**  
   **`N=1024`** tends to be the **hardest** for CK. **`N=4096`** is where **CK shows up most** in all three files — same ranking, different absolute counts.

4. **`M`**  
   Differences between **M=1**, **16**, and **32** are **smaller** than **`O`** and **`O/K`**; there is a **slight MLIR lean** in material-win counts, not a sharp “always pick this M” rule.

5. **Effect of split count**  
   **More splits → far more neutrals → fewer definitive calls.** When calls happen, **MLIR’s share of non-neutral wins rises** (about **61% → 66% → 76%**). So **more split-KV parallelism makes the two implementations look closer in absolute time**, but **when they differ meaningfully, MLIR still wins more often** in this benchmark family.

---

## Caveats

- Conclusions are **specific** to `benchmark_splitkv_ck_vs_mlir_*` and the chosen **±5%** tie band.
- They describe **this suite** and split settings, not every kernel, shape, or GPU configuration.
