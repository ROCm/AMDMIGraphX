# Migration plan: adopting `split_sym_dim` for the symbolic specialization path

**Created:** 2026-09-02 · **Last updated:** 2026-09-17 (performance resume; 15% gate failed)
**Status:** **Paused.** Do not merge into `unify-llm-refactor`. Scratch work is on
`ssd-prefill-decode` (from `origin/split_sym_dim` @ `028ad54c1`). Push that branch when ready;
this document is the handoff.
**Branches:** `unify-llm-refactor` @ `f4870391d` (ours, **untouched**) ·
`origin/split_sym_dim` @ `028ad54c1` (theirs, PR [#5123](https://github.com/ROCm/AMDMIGraphX/pull/5123)) ·
`ssd-prefill-decode` (this investigation) · `origin/develop` @ `76cde7026` (static baseline)
**Evidence:** `tmp/ssd-plan/` · `tmp/ssd-perf/20260915-161142/` ·
`tmp/ssd-rocmlir/20260915-174153/` · `tmp/ssd-develop/20260915-233856/` ·
`tmp/ssd-perf/20260917-195553/` (M0) · `tmp/ssd-perf/20260917-210521/` (M6 gate)

## Where things stand (paused 2026-09-15)

Product decision from this round still holds: unification should adopt `split_sym_dim`'s
pad-to-optimal contract (decode at 1, other in-range lengths pad to MAX / next optimal), not
keep exact `split_sizes` matching. `unify-llm-refactor` was never edited.

Work happened only in the scratch tree (`tmp/ssd_wt`, now branch `ssd-prefill-decode`). Parser
and op transplants from ours plus several `split_sym_dim` / GPU-pipeline fixes got SmolVLM2-500
through **unified** GPU compile *with* fused `group{tag: kv_cache_attention}` / `rock.attention`.
That is further than this plan's 2026-09-08 verdict ("cannot compile an LLM"). It is **not**
far enough to merge.

### What worked

| Step | Result |
|---|---|
| Parser / symbolic-attribute transplant (`dim_ops`, GQA, rotary, MatMulNBits, …) | SmolVLM2 parses on their tree |
| Topo-order blocks by walking through unplanned producers | Clears original blocker 2 (`concat in block 3`) |
| 1-input symbolic `multibroadcast` freezer | Clone bodies freeze |
| Export unplanned clone outputs (`concat_past_present` KV buffers) | Parent `resolve_replacement` no longer sees `slice in block 3` |
| Inline `normalize_symbolic_reshapes` (1-arg symbolic reshape → operand form) | Blocker 1 no longer needs a separate adapter pass |
| Freeze 2-input reshape to 1-input packed dims in clones | Stops later reshape walks from inventing 66-D all-ones layouts |
| Skip high-rank all-ones in `simplify_reshapes` | Same 66-D class after split |
| Skip dynamic reshape in `layout_convolution::score` | Parent residue no longer throws `lens()` |
| Unique `fuse_attention` submodule names (`<parent>:attnN`) | Clone seq=64 and seq=1 no longer collide on `attn0` |
| Keep int32 pad/extent math *out* of the fused body; convert integer `where` masks to bool in the parent | `rock.attention` traces QK gemm instead of a truncate / `linalg.generic` add |
| Rewrite packed-but-noncanonical literal strides in `prepare_mlir` | First static-compile abort (non-splat `0x1600x1` strides) |

Unified SmolVLM2 GPU compile succeeds. Fused kv-cache attention is present (32 groups on this
model). Driver: `--enable-symbolic` with
`--dim-param @sequence_length '{min:1,max:64}'` plus the usual batch / past / total bindings.

### Performance (SmolVLM2-500, gfx942, `-n 100`)

Compare **Total time**, not Rate. Prefill unified originally used `--batch 64`, which inflates
Rate (64 × inferences) and may `to_static(64)` leftover dyn dims.

| Workload | Unified `split_sym_dim` (scratch) | Static on `origin/develop` @ `76cde7026` |
|---|---|---|
| Decode, seq=1, `inputs_embeds {1,1,960}` | **31.26 ms** (32 inf/s) | **2.64 ms** (379 inf/s) |
| Prefill, seq=64, `inputs_embeds {1,64,960}` | **359.64 ms** (with `--batch 64`) | **2.59 ms** (386 inf/s) |

The 15% gate was: unified may be no more than 15% slower than the equivalent static mxr.
Decode is ~**12×** slower. Prefill is not a fair apples-to-apples number (batch-64 unified;
a batch-1 rerun of that mxr GPU-faulted). Either way the gate fails. Llama-2-7b awq int4 was
**not** run.

Static decode vs static prefill both ~2.6 ms on develop is consistent with a 960-wide model
that is launch-latency bound on MI300: instruction time is ~7 ms in both cases, GPU overlap
brings the reported total down. The graphs are real (32-layer gemm / attention / KV concat,
outputs `{1,1,49280}` vs `{1,64,49280}`).

### What did not work

**Scratch cannot compile the fair static mxrs.** Same ONNX, `--dim-param @sequence_length 1`
(or 64), no `--enable-symbolic`, on this branch:

```
run_high_level_pipeline: Invalid MLIR created:
  'tosa.reduce_sum' op can't trace the reduction output to a kernel result
  tensor<1x1x960xf32> -> tensor<1x1x1xf32>
```

SIGABRT 134. The kernel is a mega fusion
(`mlir_convert_convert_dot_add_…_reduce_sum_…`: 6 dots, 3 reduce_sums, RMSNorm+MLP+attention).
Unified clones never emit that module, which is why unified compile succeeds and static does
not.

A clean rebuild of the pinned rocMLIR (`ROCm/rocMLIR@c35e77b`, `BUILD_FAT_LIBROCKCOMPILER=On`)
and relink of the scratch driver did **not** change the abort. The image already had that
commit (package dated 2026-09-14). This is a legalize failure in that pin against the static
fusion shape, not a stale binary.

`is_module_fusible` try/catch, GEG disable, and jit split-fallback were tried and **reverted**
on this branch: they do not catch the rocMLIR abort (it is not a C++ exception).

**`origin/develop` static compile of the same model succeeds** (~32 s, ~695 MB mxr). So the
static graph is legal for current develop fusion + the same rocMLIR pin; scratch
`fuse_mlir` / residual parent dynamics produce a module rocMLIR cannot legalize.

### Why pause

1. **Latency.** Unified decode is an order of magnitude off the develop static baseline. That
   is not a merge candidate for prefill/decode unification even if compile is green.
2. **No fair static baseline on the candidate pass.** Scratch static compile is blocked by
   rocMLIR; develop static is a different fusion pipeline.
3. **The pass still is not merge-ready as-is.** Blocker 2 is worked around in our fork
   (topo-order, export, 1-input freeze), not fixed upstream. 2-input reshape still never
   specializes; the parent keeps a large dynamic residue (`get_tuple_elem` / `dyn_slice` /
   `reshape`). A `select_module` metadata-cache hand-merge landed on 2026-09-17 and is
   **not** enough; see [Performance resume](#performance-resume-2026-09-17).
4. **`unify-llm-refactor` remains the working unification path**
   (`split_single_dyn_dim` + `split_sizes`).

### Resume later

- Upstream: concat/block wiring + reshape representation on PR #5123, rather than carrying
  a long-lived fork of `split_sym_dim.cpp`.
- Scratch/static: stop the mega `tosa.reduce_sum` fusion (or wait for a rocMLIR pin that
  legalizes it) so unified vs static can be timed on the *same* compiler.
- The 15% gate was re-run on 2026-09-17 and still fails (~3.6× decode; prefill abort).
  Do not coalesce blocks and do not run Llama until that changes.

Local artifacts stay under `tmp/` (not in git). `unify-llm-refactor` is clean of this work.

---

## Performance resume (2026-09-17)

Stay on `ssd-prefill-decode`. Do **not** coalesce blocks. Do **not** run Llama until Smol
decode passes the gate. `unify-llm-refactor` was not edited.

### M0 (no code) — `tmp/ssd-perf/20260917-195553/`

gfx942, `MIGRAPHX_DISABLE_MLIR=1` (unified MLIR still aborts on `tosa.reduce_sum` / packed
strides). Compare **Total time**, `-n 100`, `--fill1 attention_mask`, batch 1.

| Workload | This branch unified | develop static (`tmp/ssd-wt-develop`) |
|---|---|---|
| Decode, `inputs_embeds {1,1,960}` | **33.73 ms** (instr. 57.21 ms) | **8.18 ms** (instr. 11.28 ms) |
| Prefill, `inputs_embeds {1,64,960}` | not timed this pass | not re-run this pass |

The 15% gate on this box is **8.18 × 1.15 ≈ 9.41 ms**. The older 2.64 ms develop-static
number from 2026-09-15 did not reproduce (that run had MLIR attention fusion; this pass
does not). Instruction time still overlaps the wall on MI300 (static 11.28 ms → 8.18 ms
total). The comparison is trusted as a same-machine Total-time gate, not as “same fusion.”

Diagnosis (agreed M1-first):

- 4 `select_module`s in main; decode executes the `*_0` (seq=1) clones (`{1,1,1600}` gemms
  timed, `{1,64,…}` gemms `-nan`).
- `select_module` 20.47 ms / 4 (36% of instruction time) + `get_tuple_elem` 8.55 ms / 389.
- Parent residue: 289 `dyn_slice`, 353 `get_tuple_elem` in main, 1251 main instructions.
- Clones are mostly static; 4 leftover `dyn_slice`/`eval_expr_from_shape` in clone-like
  modules. No `rock.attention` with MLIR off.

### Ports that landed (M1–M5)

- **M1** `select_module` metadata cache, discriminator indices, `argument::get_sub_object`,
  positional inner eval. Kept `is_compatible_lens`. Did **not** port `try_fast_select_eval`.
  Output routing is return-slot mapping, not unify’s name-order pack (needed for
  `split_sym_dim`). Fused GPU tuple `#output_` params map each `get_tuple_elem` to its
  return index rather than a consecutive run from the first alias.
- **M2** `replace_allocate` skip for `get_tuple_elem` of aliased tuple allocations.
- **M3** `find_const_eval_expr_from_shape`; leave symbolic `output_dyn_shapes` in place.
- **M4** `fuse_attention` captured-literal order / inlinable constants / 1-D iota. Unique
  attn names kept. No `rock.attention` in the nomlir histogram (MLIR off).
- **M5** runtime identity skip in `dyn_slice::compute` when the output shape already
  matches the input. Compile-time skip of `add_output_slice` was **not** taken (unsafe for
  mixed clones).
- Follow-on (not in the original milestone list, required to even eval): GPU lowering does
  not `copy_from_gpu` host-resident slice bounds (`eval_expr_from_shape` or `can_eval()`).
  Folding `starts={0}` to `@literal` without that skip was 289 GPU→host syncs and **212 ms**
  decode.

### M6 gate — `tmp/ssd-perf/20260917-210521/`

Same protocol, same box, nomlir mxr.

| Workload | Unified after M1–M5 | Gate (1.15 × 8.18 ms) | Result |
|---|---|---|---|
| Decode | **29.22 ms** (min 24.12, median 29.80; instr. 50.57 ms) | **9.41 ms** | **Fail** (~3.6×) |
| Prefill seq=64 | abort | — | **Fail** |

Decode vs M0 33.73 ms is a small win. `select_module` is still 18.05 ms / 4 (36%).
`get_tuple_elem` 5.99 ms / 389. `dyn_slice` 2.17 ms / 289. `hip::copy_from_gpu` is gone.
Main still has 1123 instructions, 289 `dyn_slice`, 353 `get_tuple_elem`.

Prefill (`--input-dim @inputs_embeds 1 64 960`) selects `*_1` clones then throws:

```
SELECT_MODULE: output buffer for "main:split_sym_dim_3_1:#output_:00096"
holds 98560 bytes but the selected submodule writes 6307840
```

`#output_:00096` is `{1,64,49280}` half (prefill logits). 98560 B is `{1,1,49280}`. The
parent tuple slot is still the decode-sized shape even though `output_dyn_shapes` lists
`{1, #split_sym_dim_sequence_length_target[1..64], 49280}`.

### Consult (stop here)

- **CPU vs GPU:** decode wall is still dominated by four `select_module` inner evals plus
  hundreds of parent `get_tuple_elem` / `dyn_slice`. GPU gemm time (~5 ms) is not the gap
  to 8.18 ms.
- **Residual parent histogram:** 4 `select_module`, 289 `dyn_slice`, 353 `get_tuple_elem`,
  128 `reshape`, 65 `gpu::dynamic_code_object_op`. Clones look static; the parent pad
  contract is the leftover launches.
- **Static baseline:** 8.18 ms Total / 11.28 ms instruction on this box is the gate, not
  the 2.64 ms 2026-09-15 figure (that had MLIR). Unified nomlir vs static-with-MLIR is not
  a same-compiler comparison; even so, 29 ms vs 9.41 ms is not close.
- Next levers that are **out of scope** until you say otherwise: block coalescing; turning
  MLIR back on; skipping parent `dyn_slice` at compile time; Llama.

---


## 2026-09-15 addendum (investigation log)

Two commits landed since `4d10b1656`: `update block discovery` and `update clone metadata prep`.
They rewrite the machinery that produced blocker 2, plus one new unit test
(`split_sym_dim_retries_affected_connected_edge`). Combined-tree re-measure on the new tip
(our parser + reshape adapter, their `parse_reshape` kept):

| Model | parse | compile |
|---|---|---|
| SmolVLM2-500M | yes | **same blocker 2:** `concat in block 3` `{1, sequence_length[1..64], 64}` |
| Llama-2-7b awq int4 | yes | **same blocker 2:** `concat in block 3` `{1, sequence_length[1..64], 128}` |

Logs: `tmp/ssd-plan/20260915-combined/{smol,llama}.log`. Adapter still clears blocker 1
(`lens()`). Blocker 2 is bit-for-bit the 2026-09-02 / 2026-09-08 failure.

Product decision recorded 2026-09-15: unification should adopt their pad-to-optimal contract
(decode at 1, other in-range lengths pad to MAX / next optimal), not keep exact `split_sizes`
matching. That did not, by itself, change readiness.

Our branch no longer has parse-time `unify_prefill_decode`; compile-time
`split_single_dyn_dim` + `split_sizes` *is* the unification path (`40d477885`).

### Debug of blocker 2 (scratch only; `unify-llm-refactor` untouched)

Concat-in-block-3 is not a missing `analyze_concat` case. Rotary concat is on a
**static** last axis; `sequence_length` is only parallel. The consumer is a
**2-input reshape** (`block=none`) that `can_specialize` never takes, so concat
never coalesces into the giant attention block.

| Attempt | Result |
|---|---|
| Topo-sort on `block.ops` IR edges | `edges=0` — reshape is unplanned, so concat↛consumer is invisible |
| Topo-sort on clone-body **boundary sources** | `edges=0` — boundary is the reshape, not concat |
| Topo-sort walking **through unplanned producers** | `edges=4` (`1→0, 1→3, 2→1, 3→0`); concat specializes **before** the consumer |

After that ordering, compile moved. Additional fixes that landed on `ssd-prefill-decode`:

1. **1-input symbolic `multibroadcast`.** `is_symbolic_broadcast` required
   `ninputs >= 2`, so absorbed 1-arg `out_dyn_dims` broadcasts had no freezer
   (`clone body is not fully static`). Same class as 1-arg reshape. Relaxed to
   allow the 1-input attribute form.
2. **Export absorbed graph outputs.** `concat_past_present` is unplanned, output
   is static (KV cache buffer), so `absorbable_dependency` pulls it into the
   clone body, but `find_clone_body` only exported **planned** sources. Parent
   `resolve_replacement` then hit `slice in block 3` used by
   `concat_past_present`. Exporting body ops with consumers outside the body
   (including returns) unblocks this. This op is **ours** (parser transplant).
3. **Unique attention submodule names, bool masks, high-rank reshape guards**
   (see “What worked” above). Skipping `fuse_attention` on `split_sym_dim`
   clones was only a waypoint; the landed path **keeps** kv-cache fusion.

`split_sym_dim` itself completes on SmolVLM2-500M. It still leaves residual dynamic ops in
the parent (`get_tuple_elem` / `dyn_slice` / `reshape` tagged `sequence_length` /
`#split_sym_dim_sequence_length_target`). That residue is partly by design (pad-to-optimal
dispatch) and partly because 2-input reshape still never specializes.

Logs: `tmp/ssd-plan/20260915-combined/` · unified mxr/perf `tmp/ssd-perf/20260915-161142/` ·
develop static `tmp/ssd-develop/20260915-233856/`.

---

---

## 1. Executive summary

`split_sym_dim` is a stronger specialization design than the symbolic extension we added to
`split_single_dyn_dim`, and we should still adopt it eventually. **Migration is paused
(2026-09-15).** Scratch unified compile of SmolVLM2 now works, with kv-cache fusion, but
decode is ~12× slower than develop static and scratch cannot compile the matching static
mxrs. See “Where things stand” above. The 2026-09-08 findings below are the baseline this
round started from.

Three findings shape this plan, all re-verified on 2026-09-08 against the current PR tip. The first
was downgraded on 2026-09-09 and no longer gates anything; findings 2 and 3 carry the plan:

1. **Symbolic-only scope is accepted; the pipeline wiring still needs a one-line fix.** The pass
   returns early unless a parameter shape is `symbolic()`. That restriction is fine by us — we do not
   need `split_sym_dim` to cover plain range-dynamic dimensions. Separately and independently, their
   `target.cpp` *replaced* `split_single_dyn_dim` rather than adding alongside it, which is what makes
   `test_resize_dyn` OOM there. Keeping both entries in `dynamic_shapes_pipeline()` restores it. This
   is not a design disagreement and does not gate migration.
2. **The two efforts sit in different layers and compose.** Symbolic *reach* (getting real LLM
   exports to parse) is entirely our branch's contribution; *specialization quality* is entirely
   theirs. Our parser fixes drop into their tree, build clean, and take their symbolic reach from
   0/2 to 2/2 on our target models.
3. **Their pass still cannot compile an LLM.** Two defects block it. Both were reported on
   2026-09-02 and both **reproduce bit-for-bit** at the current tip.

The payoff for adopting, once ready, is roughly **1200 deleted lines**, most valuably backing an
invasive `to_static` virtual out of the core `operation` type-erasure interface.

---

## 2. Background

Both branches descend from `core_symbolic_evaluation` and solve specialization differently.

| | ours (`split_single_dyn_dim` + `split_sizes`) | theirs (`split_sym_dim`) |
|---|---|---|
| Range-dynamic dims | yes (original dynamic-batch feature) | no — early return (accepted by design) |
| Symbolic dims | yes, via reflection (`to_static`/`sym_substitute`) | yes, via 16-family op registry |
| Clone granularity | whole module | block-local subgraphs |
| Unlisted runtime size | throws `no compatible submodules found` | pads to next optimal, masks |
| Multiple symbolic dims | one | cartesian product, `max_clones = 64` |
| Size selection | compile-time `split_sizes` backend option | parse-time `optimals` + interval min/max |
| Literal duplication | dedup in `promote_literals` (gated) | dedup in `promote_literals` (ungated) |
| Size | ~300-line pass | 2557-line pass + 2303-line test |

Both branches independently added literal deduplication to `promote_literals` — convergent evidence
that the duplication problem is real and that this is the accepted remedy.

---

## 3. Evidence

### 3.1 What changed at the 2026-09-08 tip

Three commits since the previous review (`cbb4677fd..4d10b1656`): `WIP - style`, `cleanup - WIP`,
`update coalesce algo`. Net effect is +108 lines in the pass (2449 → 2557), +6 in
`eval_expr_from_shape.hpp`, and test updates.

Two observations about branch state rather than content. The branch is now **9 commits behind
develop** (it was fully caught up on 2026-09-02), and all three commit subjects are still
self-labelled WIP.

`test/split_sym_dim_test.cpp` changed by 142 lines but the case count stayed at **53**, so the test
edits updated existing expectations for the coalesce change rather than adding coverage. None of the
53 covers a full kv-cache LLM topology.

### 3.2 Their pass is healthy in isolation

`test_split_sym_dim_test` at `4d10b1656`: **53 cases, all passing** on a pristine checkout. Padding
and masking correctness is deliberately covered at IR level (`keeps_softmax_mask_off_contract_axis`,
`preserves_compound_mask_extent`, `freezes_fixed_roots_in_mask_extents`,
`staticizes_llama_attention_chain`, `keeps_literals_in_clones`).

### 3.3 Reach is ours, not theirs

Identical `--enable-symbolic` invocations, same bindings, both drivers:

| Model | ours | split_sym_dim |
|---|---|---|
| SmolVLM2-500M ORT GenAI | parses | fails: `lens() called on a dynamic shape` |
| Llama-2-7b awq int4 | parses | fails: `lens() called on a dynamic shape` |

Both failures occur at **parse time**, under `migraphx-driver read`, before any pass runs. Our
front-end delta is 435 lines across 9 files; theirs is small and touches different files.

### 3.4 Range-dynamic: a scope choice we accept, and a separable wiring bug

These are two different things, and it is worth keeping them apart.

**The scope choice.** `has_symbolic_param` is unchanged at the new tip:

```cpp
bool has_symbolic_param(const module& m)
{
    auto param_shapes = m.get_parameter_shapes();
    return any_of(param_shapes, [](const auto& p) { return p.second.symbolic(); });
}
```

Requiring a symbolic parameter is **fine**. Every dynamic dimension in our target models is symbolic,
so a symbolic-only pass loses us nothing, and asking `split_sym_dim` to also grow range-dynamic
support would enlarge an already 2557-line pass for no benefit to this work.

**The wiring bug.** The entire `target.cpp` delta against develop is a one-line swap:

```diff
-            enable_pass(disabled(MIGRAPHX_ENABLE_FULL_DYNAMIC{}), split_single_dyn_dim{}),
+            split_sym_dim{},
```

`split_single_dyn_dim.cpp`, its header, its CMake entry, and all 8 test files that exercise it are
still present on their branch — only the GPU pipeline call site was removed, orphaning them.
`test_resize_dyn`'s input is `{[1..2], 1, 3, 3}`, range-dynamic with no symbol, so nothing specializes
it and the output allocation OOMs on `{[0..18446744073709551615], ...}`. Measured, not inferred: it
fails on their branch and passes on ours.

The fix is to keep both passes in the list rather than substituting. Because `split_sym_dim`
early-returns without a symbolic parameter and `split_single_dyn_dim` is the only thing that handles
range-dynamic, the two are close to mutually exclusive by construction. Note that the swap also
dropped the `MIGRAPHX_ENABLE_FULL_DYNAMIC` opt-out, which on develop lets a user *disable* splitting
and stay fully dynamic; whether `split_sym_dim` should honour the same knob is a separate decision.

### 3.5 Blocker 1 — symbolic-attribute reshape (unchanged)

A combined tree (their branch + 13 files from ours) **builds clean**, and both models then parse
symbolically. Compilation fails in `analyze_shape_transform`. Root cause:

```cpp
// src/shape_transform_descriptor.cpp
static std::vector<std::size_t> compute_dims(const operation& op,
                                             const std::vector<std::size_t>& idims)
{
    shape s{shape::float_type, idims};    // static shape built from max_lens()
    return op.compute_shape({s}).lens();  // throws if the op returns a dynamic shape
}
```

Our IR contains 192 `reshape[dims={1, sequence_length[1..64], ...}]` instructions — symbolic dims in
the **operator attributes**. Their pass only understands symbolic reshapes whose target arrives as an
**operand**; their `parse_reshape.cpp` change is a *precondition* for the pass, routing symbolic
reshapes through `eval_expr_from_shape` → `allocate` → two-input `reshape`.

Our 192 originate from `parse_group_query_attention.cpp` and `rotary_embedding.cpp`, which build
reshapes directly and bypass `parse_reshape`. Op counts carrying symbolic attributes: 515
`multibroadcast` (handled), 192 `reshape` (**unhandled**), 64 `dynamic_range` (handled).

**Re-measured 2026-09-08:** the `reshape` branch of `analyze_shape_transform` is structurally
identical and the failure reproduces exactly.

### 3.6 Blocker 2 — block wiring (unchanged)

The 25-line adapter in `tmp/ssd-plan/normalize_symbolic_reshapes.txt` fully clears blocker 1. The
pass then fails identically on both models, at both tips:

```
SPLIT_SYM_DIM: block dependency was not specialized before use:
  concat in block 3 shape half_type, {1, sequence_length[1..64], 64}    <- SmolVLM2
  concat in block 3 shape half_type, {1, sequence_length[1..64], 128}   <- Llama-2-7b
```

Same op, same block index, same shapes on 2026-09-02 and 2026-09-08 — the rotary-embedding concat on
the symbolic axis. Instrumenting the "block frame could not be built" path showed it **never fires**,
so this is not an unsupported operator bailing out: `concat` is fully supported by their registry. An
instruction assigned to block 3 is reachable from the module returns but never had a replacement
registered when its block was wired into the `select_module`. The defect is in
`find_clone_body`/`wire_select_module`.

One behavioural difference worth noting: the pass now reaches the failure in ~11 s of work rather
than ~150 s, consistent with `update coalesce algo` changing block discovery. The outcome did not
change.

### 3.7 Integration surface

`git merge unify-llm-refactor` into their tip still yields **10 conflicting files, 5 in `src/`**:
`fuse_attention.cpp`, `op/select_module.hpp`, `onnx/parse_reshape.cpp`, `promote_literals.cpp`,
`targets/gpu/target.cpp` (plus `CHANGELOG.md` and 4 test files). Unchanged from 2026-09-02.

`select_module.hpp` is the hard one. Our +394 lines add dispatch-metadata caching (decode latency);
their +47 add compatible-lens matching (required by padding) and alias-based output mapping. Both are
load-bearing and they touch the same functions.

---

## 4. Scope

**In scope:** replace the *symbolic extension* of `split_single_dyn_dim` with `split_sym_dim`.

**Explicitly retained:** `split_single_dyn_dim` itself, for range-dynamic shapes (§3.4). Not because
this work needs it — it does not — but because dynamic-batch specialization is a shipped MIGraphX
feature exercised by 8 test files across `resize`, `onehot`, `nonmaxsuppression`, and the C API.
Retaining it costs one line in `target.cpp`, so there is no reason to force the question.

**Out of scope:** `unify_prefill_decode`. Nothing on their branch replaces it, and with symbolic
reach still limited it remains the only path for stock HF/Optimum exports.

---

## 5. Readiness assessment (2026-09-08, revised 2026-09-15)

**Verdict: paused. Not ready to merge.** Scratch unified compile of SmolVLM2 is past the
2026-09-08 blockers; latency and static-compile-on-this-pass are not.

| Criterion | Status |
|---|---|
| Compiles SmolVLM2 end-to-end | **unified yes** (scratch `ssd-prefill-decode`); **static no** (rocMLIR `tosa.reduce_sum`) |
| Compiles Llama-2-7b end-to-end | **not attempted** this round (15% gate failed on Smol) |
| Own test suite green | yes — 53/53 |
| Symbolic core aligned with develop | **drifted** — now 9 commits behind |
| Numerics validated under padding | **not attempted** — blocked by the above |
| PR open for review | yes |
| ~~Preserves range-dynamic specialization~~ | no longer a migration criterion (§3.4) |

Accepting the symbolic-only scope removes one of the three original findings from the critical path.
It does not move the verdict, because the verdict was never driven by that finding: it is driven by
the fact that the pass cannot compile either target model, and that both failures are in code we do
not own.

Three commits of active work landed and **neither remaining blocker moved**. That is expected rather
than alarming — the commits are style, cleanup, and a coalescing-algorithm change, none aimed at
these defects — but it means the gap is unchanged after a week.

**What would change the verdict (2026-09-15):** unified vs *same-compiler* static decode/prefill
within 15% Total time, after scratch static compile is unblocked (or develop-equivalent fusion
is used without the mega `tosa.reduce_sum` module). Concat wiring and reshape representation
should land upstream on PR #5123 rather than remaining a fork.

### Recommended action now

The open PR is the right venue for these, in this order:

1. **`concat` block-wiring defect** — the one that actually blocks us, with the two-command
   reproducer in Appendix A.
2. **Reshape representation gap** — with the working adapter as a reference, framed as a design
   question about which spelling is canonical rather than as a bug report.
3. **`split_single_dyn_dim` dropped from `dynamic_shapes_pipeline`** — worth raising because it
   orphans a shipped feature and the 8 test files covering it, but as a one-line note, not an
   objection. We have no stake in the outcome.

---

## 6. Phases

### Phase 0 — Land what is valuable regardless (unblocked, start now)

None of this depends on which split pass wins, and it is what unblocks *their* branch too.

- **Parser and op fixes:** `parse_matmulnbits`, `parse_group_query_attention`, `rotary_embedding`,
  `parse_shape`, `parse_squeeze`, `parse_compare_op`, symbolic paths in `onnx_parser.cpp`, and the
  5-line `concat_past_present` dynamic-shape fix. Measured effect: their tree goes from 0/2 to 2/2 on
  symbolic parse. Verified twice to build cleanly when transplanted.
- **`promote_literals` deduplication:** both branches converged on this. Land **one** version —
  prefer our gated variant (preserves behaviour for `if`/`loop` branches), but replace the linear
  scan in both with a content hash. Both current implementations do an O(n²) full-buffer comparison,
  a latent compile-time risk on a 7B model with many identically-shaped weights.

### Phase 1 — Unblock their pass

- **Report blocker 2 on the PR** rather than debugging their pass ourselves.
- **Settle the reshape representation.** This is a design decision, not a bug fix.
  - **Recommended:** adopt their operand form as canonical, changing the handful of call sites in
    `parse_group_query_attention.cpp` and `rotary_embedding.cpp`. This is the option that *earns* the
    Phase 4 deletions, because it removes our need for symbol substitution in op attributes.
  - **Alternative:** teach `analyze_shape_transform` to accept symbolic attributes via our
    `to_static`. More general, but locks in `sym_substitute` permanently.
  - `tmp/ssd-plan/normalize_symbolic_reshapes.txt` is a working reference, verified at two tips. As a
    *permanent* pass it would paper over the split rather than close it.

### Phase 2 — Correctness and performance gates

Do not adopt until all pass. Baselines come from work already completed on our branch.

| Gate | Baseline / criterion |
|---|---|
| SmolVLM2 + Llama compile through `split_sym_dim` | Smol unified yes on `ssd-prefill-decode`; Smol static no; Llama not run |
| Numerics vs ref **at a padded length** | never validated; the key unproven risk |
| Decode / prefill latency | Develop static 2026-09-15: 2.64 ms / 2.59 ms (SmolVLM2). Unified scratch: 31.26 ms decode. Older unify-llm-refactor baseline was 3.02 / 6.11 ms. |
| Weight-literal footprint | 0.68 GB (deduped) vs 1.36 GB (unfixed) |
| Dynamic-shape suite incl. `test_resize_dyn` | passes today with `split_single_dyn_dim` retained |
| Full GPU verify suite | 3414 tests, 0 failures today |

The padded-length numerics gate matters most: `build_clone` copies literals into each clone by design
(`split_sym_dim_keeps_literals_in_clones`), so the footprint gate must be *measured*, not assumed,
and padding correctness has only ever been checked at IR level.

### Phase 3 — Pipeline integration

- **`target.cpp`:** list both passes rather than substituting — `split_sym_dim` for symbolic
  parameters, `split_single_dyn_dim` for range-dynamic. They are near-disjoint by construction
  (§3.4), so this is additive. Two things to settle deliberately rather than inherit: whether
  `split_sym_dim` should honour the `MIGRAPHX_ENABLE_FULL_DYNAMIC` opt-out that the swap dropped, and
  what should happen for a model carrying *both* a symbolic and a range-dynamic parameter, where both
  passes would fire. No model in scope does this, so a diagnostic is likely sufficient.
- **`select_module.hpp`:** hand-merge, not conflict-resolve. Keep our dispatch-metadata caching *and*
  their compatible-lens matching and alias-based outputs, with tests covering both latency and
  padding dispatch. Highest-risk edit in the migration.
- **Ordinary merges:** `fuse_attention.cpp`, `simplify_dyn_ops.cpp`, `parse_reshape.cpp`.
- **User-facing knob:** `split_sizes` (compile-time backend option) gives way to `optimals`
  (parse-time). Their pass reads `d.get_optimals()` and always inserts the interval min and max, so
  `{1, MAX_SL}` yields the decode/prefill split with no options set. Both branches already accept
  `--dim-param "@seq" "{min:1,max:128,optimals:[64,128]}"`, so no new plumbing is needed. Decide
  whether `split_sizes` survives as a compile-time way to inject optimals for pre-parsed `.mxr`.

### Phase 4 — Delete

Only after Phase 2 gates pass.

| Removed | Lines |
|---|---|
| `sym_substitute.cpp` / `.hpp` | 264 |
| `operation::to_static` virtual + generated type-erasure boilerplate | ~89 |
| `broadcast` / `multibroadcast` `to_static` overrides | 28 |
| Symbolic half of `split_single_dyn_dim` | ~324 |
| `sym_substitute_test` + symbolic `split_single_dyn_dim_test` cases | ~500 |
| **Total** | **~1200** |

The line count understates the value: the most significant item is removing a virtual from the core
`operation` interface, which currently affects every operation and requires `make generate` to
regenerate.

**Honest accounting:** our branch's delta shrinks, but if `split_sym_dim` has not merged, the
codebase *grows* by 2557 pass lines plus 2303 test lines that we would then own. The simplification is
unambiguous only in the rebase-onto-develop scenario.

---

## 7. Risks

- **Schedule coupling.** Phases 2–4 depend on a defect fix in someone else's branch, on their
  timeline. One week has passed with no movement on it.
- **Maintenance transfer.** If the pass merges, we inherit a large component. Smaller concern than
  before: the dynamic-batch regression is one line of pipeline wiring, not a property of the pass.
- **`select_module` merge.** The one place where a mistake costs correctness *and* latency at once.
- **Padding waste.** A sequence of 300 against optimals `{1, 1024}` does ~3.4× the necessary work —
  correct, but possibly unacceptable. Our current `split_sizes` throws instead.
- **Clone blowup.** `max_clones = 64` caps a cartesian product that grows quickly beyond one
  symbolic dimension.
- **Unvalidated numerics** under padding for GQA with a local window.
- **Renewed core drift.** The branch went from 0 to 9 commits behind develop in a week.

---

## 8. Open questions

1. Which reshape representation becomes canonical (§6 Phase 1)?
2. Is the removal of `split_single_dyn_dim` from `dynamic_shapes_pipeline` intentional, or an
   oversight? Either way the fix is the same one line; this only affects how it is raised on the PR.
3. ~~Should `split_sym_dim` absorb range-dynamic dimensions?~~ **Answered 2026-09-09: no.**
   Symbolic-only is the accepted scope. `split_single_dyn_dim` keeps range-dynamic and neither pass
   grows to cover the other.
4. Does the gate on `promote_literals` dedup buy anything, or is the ungated form safe? Literals are
   immutable and `promote_literals` already hoists them to root regardless, which argues the gate is
   conservatism rather than necessity.

---

## 9. Recommended sequencing

**Paused 2026-09-15.** Do not start Phases 2–4. `unify-llm-refactor` stays on
`split_single_dyn_dim` + `split_sizes`. Scratch lives on `ssd-prefill-decode` for a later
resume (fair static baseline on this pass, then the 15% gate, then Llama). Phase 0 parser
fixes are already on that branch. Posting the concat/reshape findings to PR #5123 remains the
right upstream action when this is picked up again.

---

## Appendix A — Reproducers

Symbolic parse reach, either driver:

```bash
migraphx-driver read /models/smolvlm2-500/model.onnx --onnx --enable-symbolic \
  --dim-param '@sequence_length' '{min:1,max:64}' --dim-param '@batch_size' 1 \
  --dim-param '@past_sequence_length' 64 --dim-param '@total_sequence_length' 128
```

Blocker 1 (`lens()`), on a tree carrying our parser fixes — swap in the Llama path for the second
model:

```bash
migraphx-driver compile /models/smolvlm2-500/model.onnx --onnx --enable-symbolic \
  --dim-param '@sequence_length' '{min:1,max:64}' --dim-param '@batch_size' 1 \
  --dim-param '@past_sequence_length' 64 --dim-param '@total_sequence_length' 128 --gpu
```

Blocker 2 (`concat` wiring): same command, with the adapter from
`tmp/ssd-plan/normalize_symbolic_reshapes.txt` applied.

Pipeline-wiring regression on their branch (informational — not a migration blocker, §3.4):

```bash
./bin/test_verify test_resize_dyn
```

## Appendix B — Artifacts

- `ssd-prefill-decode` — cleaned scratch branch from `origin/split_sym_dim` @ `028ad54c1`
  (parser transplant + `split_sym_dim` wiring + attention fusion + reshape/layout/prepare_mlir
  guards). Debug prints and experimental rocMLIR try/catch probes were removed before commit.
- `tmp/ssd_wt` — worktree that held this work; may still be checked out on `ssd-prefill-decode`.
- `tmp/ssd-plan/` — earlier compile logs and the reshape adapter
  (`normalize_symbolic_reshapes.txt`).
- `tmp/ssd-perf/20260915-161142/` — unified Smol mxr + perf (`smol_unified.mxr`, 31.26 ms decode).
- `tmp/ssd-rocmlir/20260915-174153/` — rebuild of rocMLIR `@c35e77b`; did not fix static abort.
- `tmp/ssd-develop/20260915-233856/` — develop static mxrs + perf (2.64 ms decode, 2.59 ms prefill).
- `tmp/ssd_develop` — detached worktree at `origin/develop` @ `76cde7026`.
- `tmp/ssd-perf/20260917-195553/` — M0 re-measure (33.73 ms unified decode nomlir, 8.18 ms develop static).
- `tmp/ssd-perf/20260917-210521/` — M6 gate (29.22 ms unified decode; prefill abort).

`unify-llm-refactor` was not modified at any point during this investigation.
