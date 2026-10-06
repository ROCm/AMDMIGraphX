---
description: Review the changed MIGraphX code for reuse, simplification, efficiency, altitude, safety, and comment quality, then apply the fixes. Quality only — it does not hunt for correctness bugs; use /code-review for that.
allowed-tools: Bash(git diff:*), Bash(git status:*), Bash(git merge-base:*), Bash(git log:*), Bash(git rev-parse:*), Bash(git branch:*), Bash(git fetch:*), Bash(grep:*), Bash(find:*), Read, Edit, Write, Agent
---

# migraphx-simplify: Code Review and Cleanup

You are improving the quality of the changed code, not hunting for bugs. Review
it for **reuse, simplification, efficiency, altitude, safety, and comment
quality**, then fix what you find. Do not look for correctness bugs — that is
what `/code-review` is for.

## Phase 0 — Gather the diff

Determine the set of changed code under review, then hand the same diff to every
agent so they all see identical context.

The base must be the **up-to-date remote** integration branch — use
`origin/develop`, **not** the local `develop` ref. The local `develop` is
frequently stale (behind origin by several already-merged PRs), and feature
branches often merge `develop` in repeatedly. Diffing against a stale local
`develop` silently pulls already-merged upstream PRs into the diff, so you end up
reviewing and editing files that aren't this branch's work at all.

1. Resolve the base ref. Prefer `origin/develop`; if it doesn't resolve, fall
   back in order to `origin/main`, `origin/master`, then a local `develop` /
   `main` / `master` — but never default to a local branch when a
   remote-tracking ref exists. Optionally `git fetch origin develop` first if the
   network is available and you suspect `origin/develop` itself is stale.
   ```bash
   git rev-parse --abbrev-ref HEAD
   git rev-parse --verify --quiet origin/develop   # confirm the base resolves
   ```
   Use the resolved ref as `<base>` below.

2. Produce the combined diff of committed-on-this-branch **and** uncommitted
   working-tree changes versus the merge base:
   ```bash
   git merge-base HEAD <base>
   git diff $(git merge-base HEAD <base>)
   ```

3. Sanity-check the scope **before** launching agents. List the changed files and
   confirm they look like *this* branch's work:
   ```bash
   git diff --stat $(git merge-base HEAD <base>) HEAD
   ```
   If the list contains files you don't expect — unrelated ops, whole subsystems,
   or infra you didn't touch — the base is almost certainly wrong. A tell-tale
   sign is that HEAD is a `Merge branch 'develop' into <feature>` commit (its
   develop-side parent `HEAD^2` is usually the correct base). Confirm a suspect
   commit is upstream (and therefore out of scope) before trusting the diff:
   ```bash
   git merge-base --is-ancestor <commit-sha> <base>   # exit 0 => upstream, exclude it
   ```
   Re-resolve `<base>` (e.g. to `HEAD^2`) until the file list matches only the
   branch's own changes.

4. If the diff against the base is empty (e.g. you are already on the base
   branch), fall back to the working-tree changes alone:
   ```bash
   git diff HEAD
   git status --short
   ```

5. List the changed files and note which are GPU kernel headers (under
   `src/targets/gpu/kernels/`) versus general host C++ — the kernel-specific
   bullets below only apply to kernel files.

If there are no changes at all, stop and report that there is nothing to review.

## Phase 1 — Review (6 cleanup agents in parallel)

Launch **6 independent review agents via the Agent tool, all in a single
message** so they run concurrently (use `subagent_type: general-purpose`; they
need Grep/Read to inspect the surrounding codebase). Give each agent the full
diff from Phase 0, the list of changed files, and the one review angle below that
is its responsibility.

Tell every agent: assume all tools work — do not test them or make exploratory
calls without a purpose. Each agent returns its findings as a list, where each
finding has `file`, `line`, a one-line `summary`, and the concrete cost (what is
duplicated, wasted, unsafe, misleading, or harder to maintain) plus the concrete
fix (name the helper, the simpler form, or the safe abstraction to use instead).
Report only findings inside the changed code; do not flag pre-existing issues.

### 1. Reuse
Flag new code that re-implements something the codebase or standard library
already provides. Grep shared/utility modules and files adjacent to the change
before concluding something is new, and name the existing helper to call
instead.
- Flag any new function that duplicates existing functionality - Suggest the existing function to use instead.
- Flag any inline logic that could use an existing utility — hand-rolled string manipulation, manual path handling, custom environment checks, ad-hoc type guards, and similar patterns are common candidates.
- Copy-paste with slight variation: near-duplicate code blocks that should be unified with a shared abstraction
- Raw loops that should be STL `<algorithm>` or MIGraphX `<migraphx/algorithm.hpp>`
  algorithms (`transform_if`, `transform_accumulate`, `group_by`, `group_unique`,
  `adjacent_for_each`, etc.). `std::for_each` is not an acceptable substitute for
  a real algorithm, but a raw loop is the *least* preferred form: never rewrite a
  `std::for_each` into a raw `for` loop. If it is a side-effecting fold with no
  clean algorithm (e.g. folding `hash_combine` into a seed), leave it alone.
- A `contiguous` inserted before `reshape` — `reshape` already copies and handles
  non-standard input layouts, so the extra op is redundant.
- Pair/tuple member access (`auto r = f(); r.first`) where a structured binding
  `auto [a, b] = f();` fits; when a C++17 lambda needs the binding, use an
  init-capture (`[&, &x = x]`) rather than avoiding the binding.
- Manually written lexicographical comparisons — use `std::tie` or
  `std::lexicographical_compare` instead.
- Manual offset / stride / index math with mod and division — use the
  `migraphx::shape` methods for offsets, strides, and indexing instead.
- Open-coded instruction traversal where a matcher from `src/matcher.hpp` would do.
- C acquire/release handling that should use `MIGRAPHX_MANAGE_PTR`.
- Duplicated literals or magic numbers where a named constant or existing enum
  already exists.
- A new helper that duplicates one already living elsewhere — name the existing
  function.
- **Not** a duplicate: a bespoke test stub operator (e.g. `view_pass_op` in
  `test/gpu/hipgraphify.cpp`) that a `basic_ops` stub could replace behaviorally.
  Check which branch of the predicate under test each one drives (no `compute` → not
  context-free, versus context-free + aliasing); if they differ, swapping them
  silently changes what the test covers.
- (kernel files) Hand-rolled utilities that already exist: `migraphx::array`,
  `tensor_view`, `vec<T, N>` for vector types, `repeat_c` for unrolling loops,
  `uninitialized_buffer` for shared LDS, the `index` class for thread/block
  indices, and `local_stride` / `global_stride` / `local_wave_stride` for strided
  loops.

### 2. Simplification
Flag unnecessary complexity the diff adds — code that could do the same job in a
simpler, smaller, or flatter form. Name the simpler form that does the same job.
- Avoid redundant or unnecessary casts.
- Redundant or derivable state; intermediate variables that can be used directly.
- Copy-paste with slight variation that should be factored or parameterized.
- Deep nesting that an early return or restructuring would flatten.
- Dead code left behind by the change. **Not** dead code: a shared kernel utility
  (e.g. `partial` in `kernels/functional.hpp`) whose last caller this diff removed —
  the kernel functional library is kept as a toolkit, so only remove helpers the
  change itself introduced.
- Premature abstraction — a helper used only once.
- Classes masquerading as functions — a stateless struct with one public method
  *instantiated as an object* should be a free function. A `static` member function
  that happens to use no member state is **not** this: it is scoped to its class on
  purpose, so leave it where it is.
- Locally-called named lambdas (`auto used_outside = [&](...) {...};` followed by a
  call) — a function in disguise with hidden captures. Convert to a `static` free
  function taking its captures as parameters, or inline a single use. Lambdas
  *passed* to an algorithm or callback are fine.
- Static helpers that each take 5+ parameters to marshal shared state after a
  function was split — replace with a (nested) struct holding that state in value
  members, with methods that add to it; keep `context&` and other external state as
  method parameters.
- Config / builder structs around what should be 1–2 function arguments; factory
  functions that only call a constructor.
- Multiple wrapper layers with no distinct responsibility per layer — collapse them.
- Future-proofing: single-value enums, one-alternative `variant`, always-present
  `optional` returns, hooks/callbacks with no second caller.
- Unnecessary arithmetic (`+0`, `*1`).
- Defensive checks at internal boundaries and speculative error handling for
  conditions that cannot occur given caller guarantees — remove them. (This is
  distinct from assertions that document assumptions, which belong to the safety
  pass.) An `assert` at an exported function's boundary that a later internal helper
  re-asserts is **not** a duplicate: each layer documents its own contract, so keep
  both.
- Backwards-compat shims that aren't needed: renamed `_unused` parameters,
  re-exports of removed types.
- Unrequested refactors bundled into the change — flag them for separation.
- A one-line layout predicate (e.g. `is_nhwc(const shape&)`) added to a widely
  included core header such as `permutation.hpp` — inline the stride check at each
  site with a comment instead; narrow GPU/conv helpers do not belong in core headers.
- (kernel files) `::value` used to read an integral constant — it converts
  implicitly; pass the `integral_constant` through and capture it with `auto`.
  In **tests**, `::value` is read on purpose to check the stored constant: keep it
  and add a `cppcheck-suppress migraphx-AvoidNestedValue`.
- (kernel files) Do **not** flag `if constexpr` special cases that look redundant
  with the generic path (e.g. `arg_c` for `N == 0`): kernel headers are JIT-compiled
  at runtime and those branches cut template instantiations. If one lacks a comment
  saying so, add the one-line comment rather than removing the branch.
- (kernel files) Do **not** merge per-kernel JIT raw-string source templates into a
  shared `${...}` shell; the explicit per-kernel template is the preferred form.
- (kernel files) `static_cast<T&&>(x)` is a move, not a redundant cast — keep it
  and suppress `migraphx-RedundantCast`; do not introduce a `move()` helper.

### 3. Efficiency
Flag wasted work the diff introduces — computation, memory, or build cost that a
cheaper form avoids without changing behavior. Name the cheaper alternative.
- Redundant computation or repeated I/O that could be hoisted or cached.
- Independent operations run sequentially that could be combined.
- Blocking or expensive work added to startup or a hot path.
- Copies where a reference or view suffices; unnecessary `auto&&`.
- Unused `std::move` and unused casts.
- Over-broad `#include`s where a forward declaration or a narrower header would do.
- A non-template helper defined inline in a header — it drags its includes into
  every consumer. Declare it in the header (with the target's export macro) and
  move the body to the `.cpp`; only templates and trivial accessors stay inline.

### 4. Altitude
Check that each change is implemented at the right depth, not as a fragile
bandaid, and that it is written generically rather than tied to one concrete
type. Special cases layered on shared infrastructure signal the fix isn't deep
enough — prefer generalizing the underlying mechanism.
- Special cases added on top of shared infrastructure where generalizing the
  mechanism would be cleaner than another branch.
- Type-specific code that should be generic — write reusable, type-independent
  code using templates and STL algorithms rather than duplicating per type.
- Open-coded traversal where a matcher-based rewrite expresses the transform at
  the right level.
- Hand-rolled polymorphism via inheritance where MIGraphX's type erasure
  (implement the interface, no base class) fits.
- Leaky abstractions - exposing internal details that should be encapsulated, or breaking existing abstraction boundaries

### 5. Safety
Flag places where the diff gives up safety it could keep for free — unchecked
assumptions, raw memory and pointer handling, and unsafe type or bit
manipulation. Do **not** recommend adding defensive checks at internal
boundaries or speculative error handling; assertions that document an invariant
are different and are encouraged.
- Silent assumptions (a size matches, an index is in range, a pointer is
  non-null, a shape is packed/standard) — add an `assert` / MIGraphX assertion
  that documents and checks the invariant.
- Manual buffers or raw C arrays — use `std::vector` / `std::array`; in kernels
  use `migraphx::array`.
- Raw `new` / `delete` — use `std::make_unique` / `std::make_shared`; use
  `MIGRAPHX_MANAGE_PTR` for C-style acquire/release resources.
- Raw pointer arithmetic — use iterators, range-based access, or `tensor_view`;
  in kernels prefer `tensor_view` / iterators over raw pointers and the `index`
  class over raw `threadIdx` / `blockIdx`.
- Manual offset / index math that can run out of bounds — compute via
  `migraphx::shape` (host) or `tensor_view` / `shape` / `index` (kernels); slice
  a `tensor_view` / `shape` for tiling rather than hand-computing offsets, and
  use `uninitialized_buffer` for shared LDS.
- Unsafe casts — `const_cast` (especially to bypass API design), C-style casts,
  and `reinterpret_cast` that bypass the type system; fix by declaring the
  correct type at the source.
- Bit-twiddling and low-level bit manipulation written inline — encapsulate it
  behind a well-named utility function and prefer `std::bitset`.

### 6. Comments
Check that comments in the diff are correct, clear, and relevant, and remove
comment slop. Quote the current comment and give the corrected / shortened
replacement, or recommend deletion.
- Correct — the comment matches what the code actually does; flag stale comments
  left after the code changed and comments describing the wrong behavior.
- Clear — flag confusing, ambiguous, or misleading wording; rewrite to state the
  non-obvious *why* (a constraint, workaround, or invariant) in one to three lines.
  A rewrite must not grow the comment: drop parentheticals, cross-references, and
  enumerated consequences unless they are load-bearing.
- Relevant — flag comments that discuss unrelated components, files, or code not
  present in the change.
- Tombstones — flag `// removed ...` / "this used to ..." comments that describe
  history instead of the current code.
- Slop — delete redundant comments describing *what* obvious code does
  (`// increment i`, `// return result`).
- Banners — collapse multi-line docstrings / banner comments on trivial functions
  to one short line at most.

## Phase 2 — Apply the fixes

Wait for all six agents to complete. Dedup findings that point at the same line
or mechanism (the angles overlap by design — e.g. reuse and safety may both flag
a raw pointer that should be a `tensor_view`). Fix each remaining finding
directly in the working tree with Edit.

Skip any finding whose fix would:
- change intended behavior,
- require changes well outside the reviewed diff, or
- that you judge to be a false positive

— note the skip rather than arguing with it. Do not modify tests to match
changed output; if a fix breaks a test, the fix is wrong.

Finish with a brief summary grouped by angle: what was fixed (with `file:line`)
and what was skipped and why — or confirm the code was already clean.
