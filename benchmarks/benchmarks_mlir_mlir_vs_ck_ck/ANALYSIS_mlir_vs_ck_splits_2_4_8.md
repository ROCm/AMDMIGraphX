# MLIR vs CK (split-KV) benchmark analysis — splits 2, 4, and 8

This note summarizes patterns from `stats_2_splits.csv`, `stats_4_splits.csv`, and `stats_8_splits.csv` in this directory. They come from `benchmarks/scripts/analyze_ck_mlir_benchmark_csvs.py` run on `benchmark_splitkv_ck_mlir_vs_ck_ck_*_splits.csv`. The benchmark compares **MLIR** and **CK**; the stats CSV uses the column names **`ck_*`** for the CK side and **`mlir_*`** for the MLIR side (same schema as other split-KV CSVs).

**Definitions:** “Definitively faster” = outside the default **±5%** neutral band on `speedup_pct` (`ck_win` vs `mlir_win` in the stats).

---

## 2 splits (576 rows)

**Verdict:** 22.0% neutral, **55.9%** CK wins, **22.0%** MLIR wins. Among **non-neutral** rows, **~72%** favor **CK** (322 vs 127).

### Where CK is clearly faster

- **`O` small / medium:** **O=32**, **O=48**, **O=64** — **72** CK vs **0** MLIR material wins each; **O=80** still strongly **CK** (48 vs 2).
- **`N=1024`:** **140 vs 25** — largest CK lean among the three **N** slices.
- **Geometry `O < K`:** buckets **`K/O>4`**, **`2<K/O<=4`**, **`1<K/O<=2`** have **0** or very few MLIR material wins (e.g. **`2<K/O<=4`**: 80 vs 0; **`K/O>4`**: 27 vs 0).
- **`M`:** all three **M** values favor **CK** in counts (~100–113 vs ~40–45).

### Where MLIR is clearly faster

- **`O` large:** **O=192** (**2 vs 45**), **O=256** (**2 vs 64**) among material wins.
- **`O > K`:** **`2<O/K<=4`** (**15 vs 54**) and **`O/K>4`** (**0 vs 22**) — MLIR material wins concentrate in **fat-head** ratio bands.
- **`N`:** MLIR’s share is **highest** at **N=4096** (84 CK vs 53 MLIR) — CK still leads overall, but the gap is narrowest there.

### `(K, O)` grid

Small **O** cells are almost entirely **CK**; for **O ≥ 192** many cells flip to all or mostly **MLIR** material wins.

---

## 4 splits (576 rows)

**Verdict:** **13.5%** neutral, **75.9%** CK wins, **10.6%** MLIR wins. Among **non-neutral**, **CK ~88%** (437 vs 61).

### Where CK is clearly faster

- **`O=32`–`O=128`:** **0** MLIR material wins for **O=32, 48, 64** (**72 vs 0** each); **O=80** still **65 vs 0**; **O=96/128** remain CK-heavy (**58 vs 3**, **51 vs 3**).
- **`N=1024`:** **190 vs 1** — extreme CK skew.
- **`O < K` buckets:** **`K/O>4`**, **`2<K/O<=4`**, **`1<K/O<=2`** have **0** or **1** MLIR material wins in the coarse table.
- **`M`:** CK leads at every **M** (**143–148** vs **18–22**).

### Where MLIR is clearly faster

- **`O=256`:** **23 vs 39** — MLIR wins **more** material rows than CK (one of the few slices where that happens).
- **`O=192`:** still CK-heavy (**24 vs 16**) but MLIR is non-trivial.
- **`2<O/K<=4` / `O/K>4`:** MLIR material wins appear (**26** and **13** vs **44** and **9** CK) — same **large `O` / large `O/K`** story as at 2 splits, but CK still wins a large share of those cells.

---

## 8 splits (576 rows)

**Verdict:** **7.5%** neutral, **89.2%** CK wins, **3.3%** MLIR wins. Among **non-neutral**, **CK ~96%** (514 vs 19).

### Where CK is clearly faster

- **`N=1024`:** **187 vs 0** MLIR material wins — every decisive row in that slice favors **CK**.
- **`O=32`–`O=96`:** **72 vs 0** MLIR for **O=32…O=80**; **O=96** is **69 vs 0**.
- **`O < K` and `O=K` coarse buckets:** MLIR material wins are **0** or **1–3** per row in the O/K table; CK dominates every band numerically.
- **`M` / `K`:** CK leads every marginal bucket.

### Where MLIR is still visible

- **`O=256`:** **38 vs 13** — CK still ahead, but MLIR is **largest** at this **O** (only **O=128** has a single MLIR material win at **1**).
- **`2<O/K<=4`:** **67 vs 9**; **`O/K>4`:** **15 vs 4** — residual MLIR wins stay in **high `O` relative to `K`** regimes, but they are **rare** overall.

---

## Overarching patterns (2 → 4 → 8 splits)

1. **Trend vs `benchmarks_mlir_mlir_vs_ck_mlir` on neutrals**  
   In **this** suite (`ck_mlir_vs_ck_ck`), **more splits** go with **fewer** neutrals (**22% → 13.5% → 7.5%**) and **more** decisive **CK** wins (**~56% → ~76% → ~89%** overall). The two paths **separate** more clearly as split count increases, and **CK** almost always wins outside the tie band here.

2. **`O` (head width) remains the main axis for MLIR wins**  
   MLIR material wins cluster at **large `O`** (**192**, **256**) and in **`2<O/K<=4`** / **`O/K>4`**. **Small and medium `O`** stay CK-only or CK-heavy across all three split settings.

3. **`N=1024`**  
   **CK** is strongest at **N=1024** at **4** and **8** splits (including **zero** MLIR material wins at **8** splits). **N=4096** is where MLIR is **relatively** least behind at **2** splits; at **8** splits MLIR counts pick up slightly in some buckets but stay a small minority.

4. **`O < K` geometry**  
   **`K/O>4`**, **`2<K/O<=4`**, **`1<K/O<=2`** are overwhelmingly **CK** at **4** and **8** splits; at **2** splits a few MLIR wins appear in **`1<K/O<=2`** and **`O=K`**, but MLIR never dominates those bands.

5. **`M` and `K`**  
   No clean “always pick this **M**/**K**” rule for MLIR; both dimensions are mostly **CK**. MLIR shows up as **exceptions**, not as a systematic **M**/**K** story.

**Summary:** In this **MLIR vs CK** split-KV suite, **CK wins decisively most of the time**, and **more splits sharpen that advantage**. **MLIR** only finds consistent material wins in **large-head** regimes (**high `O`**, especially **`2<O/K<=4`** and **`O/K>4`**), and even there **CK** often still leads at higher split counts.

---

## Caveats

- **`ck_*`** columns = **CK** path; **`mlir_*`** columns = **MLIR** path (benchmark CSV naming).
- Conclusions are tied to this **shape grid**, **±5%** neutral rule, and **2 / 4 / 8** split counts only.
