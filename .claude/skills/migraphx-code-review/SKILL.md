---
description: "Code review the changed MIGraphX code for correctness bugs, language-specific pitfalls, C/C++ API-ABI breakage, missing test coverage, and convention violations, with a verify pass that drops false positives. The quality checklist is delegated to /migraphx-simplify rather than repeated. Effort levels low through max; --comment posts inline PR comments, --fix applies every class of finding including the quality cleanups."
allowed-tools: Bash(git diff:*), Bash(git status:*), Bash(git log:*), Bash(git show:*), Bash(git blame:*), Bash(git rev-parse:*), Bash(git merge-base:*), Bash(git branch:*), Bash(git fetch:*), Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh api:*), Bash(grep:*), Bash(find:*), Read, Grep, Glob, Edit, Agent, Skill, ReportFindings, Artifact, mcp__github_inline_comment__create_inline_comment
---

# migraphx-code-review

Usage: `/migraphx-code-review [low|medium|high|xhigh|max] [--fix] [--comment] [<target>]`

Pick the effort level from the first argument; if none is given, use the session
effort, defaulting to **medium**. Anything after the level is the review target
(a PR number, branch name, ref range, or file path).

This skill hunts **bugs and structural defects**. It does not restate the rules
that already live elsewhere — it reads them and reviews against them:

- `AGENTS.md` (tracked, canonical) and `CLAUDE.md` — build, architecture,
  coding standards, extension contracts, testing, linting, common traps.
- `.claude/skills/migraphx-simplify/SKILL.md` — the quality checklist (reuse,
  simplification, efficiency, altitude, safety, comments).

Read those files when an angle below points at them. Never paraphrase a rule
from memory: quote it from the file you read.

| Level | Candidates per angle | Verify | Sweep | Cap |
|-------|----------------------|--------|-------|-----|
| `low` | one diff pass, no angles | none | no | 4 findings |
| `medium` | 6 | 1-vote, precision-biased | no | 8 findings |
| `high` | 6 | 1-vote, recall-biased | no | 10 findings |
| `xhigh` | 8 | 1-vote, recall-biased | yes | 15 findings |
| `max` | 8 | 1-vote, recall-biased | yes | 15 findings |

Open with the stance for the level:

- **medium** — You are reviewing for **precision**: every finding you surface
  should be one a maintainer would act on.
- **high** — You are reviewing for **recall**: catch every real bug a careful
  reviewer would catch in one sitting. Catching real bugs matters more than
  avoiding false positives. Err on the side of surfacing.
- **xhigh / max** — You are reviewing for **recall** at extra-high (or maximum)
  effort: catch every real bug.  At this level, catching real bugs matters more
  than avoiding false positives — A missed bug ships. Err on the side of
  surfacing.

---

## `low` effort — one diff pass, no verify, ≤4 findings

Run only the two turns below; skip every phase that follows.

### Turn 1 — read

One tool call: read the unified diff (`git diff $(git merge-base HEAD origin/develop)`,
or the target passed as an argument). Skip test and fixture hunks (`test/`,
`*_test.cpp`, `test/onnx/*.onnx`, `test/py/`) — test changes are not reviewed at
this level. No subagents, no full-file reads.

### Turn 2 — findings

Flag runtime-correctness bugs visible from the hunk alone: inverted or wrong
condition, off-by-one, `std::size_t` underflow, null or `end()` deref where
adjacent lines show the value can be absent, a removed guard, a swallowed
exception that should propagate, wrong-variable or wrong-axis copy-paste, an
`instruction_ref` used after the module was mutated. Also flag — still from the
hunk alone — new code that duplicates a helper visible in the diff context, and
dead code the diff leaves behind.

Do **not** flag style, naming, perf, missing tests, or anything outside the
hunk. Test coverage is deliberately not reviewed at this level — it starts at
`medium` (Angle I), which reads the mirrored test files.

Report at most **4 findings**, most-severe first, in one `ReportFindings` call
with `{level, findings}` — each entry has `file`, `line`, `summary`,
`short_summary` (≤60 characters), and `failure_scenario`. If nothing qualifies,
call it with an empty findings array. Do not also print the findings as text.
If `ReportFindings` is unavailable, print one line per finding as
`path/to/file.cpp:123 — what's wrong and the concrete failure`, or exactly
`(none)`.

---

## Phase 0 — Gather the diff

Follow **Phase 0 of `.claude/skills/migraphx-simplify/SKILL.md`** — read it and
do what it says. It resolves the base against `origin/develop` (not the stale
local `develop`), takes the merge-base diff, folds in uncommitted work, and
sanity-checks the file list before any agent is launched. Do not re-derive that
procedure here.

If a `<target>` was passed, review it instead: `gh pr diff <n>` for a PR number,
the given range for a ref range, or restrict the diff to the given paths.

Then classify the changed files — the conditional angles below are gated on this:

- **languages present**: C++ host (`src/**`, `test/**`), HIP/device
  (`src/targets/gpu/kernels/**`, `src/targets/gpu/device/**`), Python
  (`tools/*.py`, `src/py/**`, `test/py/**`, `examples/**`), Bash (`*.sh`),
  CMake (`CMakeLists.txt`, `cmake/*.cmake`), MLIR-facing C++
  (`src/targets/gpu/mlir.cpp`, `fuse_mlir.cpp`), ONNX/TF parsers (`src/onnx/**`,
  `src/tf/**`), CI YAML (`.github/**`).
- **C/C++ API surface**: `src/api/**`, `tools/api/**`,
  `src/api/include/migraphx/migraphx.h`, `.../migraphx.hpp`.
- **IR surface**: `src/op/**`, passes in `src/**` with their headers,
  `src/targets/*/target.cpp`.

If there are no changes at all, stop and report that there is nothing to review.

## Phase 1 — Find candidates

Launch the angles as **independent agents via the `Agent` tool, all in a single
message** so they run concurrently (`subagent_type: general-purpose`; they need
Read and Grep). Give each agent the full diff, the classified file list, and the
one angle it owns. Each candidate has `file`, `line`, a one-line `summary`, and
a concrete `failure_scenario`.

**Core angles** (A, B, C, Languages, Quality, Conventions, Tests, Precedent)
always run.
**Conditional angles** (API/ABI, IR contracts) run when Phase 0 says their
surface was touched; at `xhigh` and `max` run the IR-contract angle regardless.

Pass every candidate with a nameable failure scenario through — finders that
silently drop half-believed candidates bypass the verify step and are the
dominant cause of misses. Do NOT let one angle's conclusions suppress
another's: if two angles flag the same line for different reasons, record both.

Tell every agent: assume all tools work — do not test them or make exploratory
calls without a purpose.

If the `Agent` tool is not available, see *Running without the Agent tool*.

### Angle A — line-by-line diff scan

Read every hunk in the diff, line by line. Then Read the enclosing function for
each hunk — bugs in unchanged lines of a touched function are in scope (the PR
re-exposes or fails to fix them). For every line ask: what input, shape, type,
target, or state makes this line wrong? Look for inverted or wrong conditions,
off-by-one, unsigned underflow, null or `end()` deref, a reference bound to a
temporary, wrong-variable copy-paste, an error swallowed where `MIGRAPHX_THROW`
should propagate.

### Angle B — removed-behavior auditor

For every line the diff DELETES or replaces, name the invariant or behavior it
enforced, then search the new code for where that invariant is re-established.
If you can't find it, that's a candidate: a removed guard, a dropped error path,
a narrowed validation, a matcher predicate that now lets extra instructions
through, a type dropped from a supported-type list, a pass removed from a
target's `get_passes()`, a deleted test that was covering a real case.

### Angle C — cross-file tracer

For each function the diff changes, Grep for its callers and check whether the
change breaks any call site: a new precondition, a changed return shape, a new
exception, a timing or ordering dependency. Also check callees: does a parallel
change in the same PR make a call unsafe? In this codebase that also means: an
op whose semantics changed must stay consistent across `ref`, `cpu`, and `gpu`
plus every parser that emits it, and a changed pass must still be correct at its
position in each target's pass list.

### Angle D — language-pitfall specialists (one agent per language present)

Do not run this as one agent. For **each language Phase 0 found in the diff**,
launch a separate finder that owns that language, and skip the languages with no
changed files. Give each one the hunks in its language plus the full diff for
context. Each hunts the pitfalls its language actually has:

- **C++ (host)** — lifetime and dangling references, iterator and
  `instruction_ref` invalidation while mutating a module, integer overflow and
  signed/unsigned mixing in index math, implicit narrowing, order-of-evaluation
  assumptions, exception safety across a resource acquisition, `std::vector<bool>`,
  copy-vs-move mistakes, ODR problems from non-inline definitions in headers,
  static initialization order.
- **HIP / device code** — wavefront-size assumptions (RDNA targets such as
  gfx1201 are wave32, not wave64), missing or mismatched barriers before LDS is
  reused, races on shared memory, out-of-bounds on the tail when vectorizing,
  alignment requirements of vectorized loads, `__shared__` sized from a runtime
  value, LDS budget overrun, arch gating (MFMA vs WMMA), a non-trivially-copyable
  type passed to a kernel, host/device synchronization mistakes.
- **Python** — mutable default arguments, late-binding closures, exceptions
  swallowed, subprocess quoting and shell injection, iteration over a mutating
  container, float formatting assumptions; in `src/py/` also pybind11 issues —
  reference lifetime versus the underlying C++ object, GIL handling around
  long-running calls, buffer-protocol shapes and strides, and a new C++ API with
  no binding.
- **Bash** — unquoted expansions and word splitting, missing `set -euo pipefail`,
  `[` versus `[[`, unguarded `cd`, glob expansion on empty matches, exit codes
  swallowed by a pipeline, plus the command-invocation rules stated in
  `CLAUDE.md`/`AGENTS.md`.
- **CMake** — a new source or header not added to its target, install, or embed
  list; a dependency declared `PRIVATE` that headers expose; an option default
  flipped; a generator expression that silently evaluates empty.
- **MLIR-facing C++** — operation-name and attribute strings that must match
  what rocMLIR expects, type and layout conversion between `migraphx::shape` and
  the MLIR module, tuning-key construction, and the fallback path when MLIR
  returns no solutions for a problem.
- **ONNX / TF parsers** — opset differences, attribute defaults that disagree
  with the spec, negative axes used without normalization, optional inputs given
  as empty names, int64 to int32 narrowing, broadcasting semantics.
- **CI YAML** — a job that no longer runs what its name claims, a matrix entry
  dropped, a step whose failure is masked.

Each specialist reports in the same candidate shape as every other angle.

### Angle E — C/C++ API and ABI auditor *(when the API surface changed)*

Review `src/api/include/migraphx/migraphx.h` (C) and `migraphx.hpp` (C++), and
the generator inputs under `tools/api/`, for compatibility and durability:

- **API breakage** — a removed or renamed function, a changed parameter list or
  return type, a changed ownership or lifetime contract, a changed error
  convention, a semantic change behind an unchanged signature. Any of these
  breaks source compatibility for existing callers; say so explicitly.
- **ABI breakage** — a struct that changed size, layout, or field order; an
  enum whose existing values were renumbered or which is passed by value where a
  new enumerator changes behavior; a changed calling convention; a function
  removed from the exported set; anything that makes an already-compiled client
  binary wrong rather than merely uncompilable. Distinguish this from API
  breakage in the finding.
- **Robustness against future ABI changes** — the C API should hand out
  **opaque handles** (`typedef struct migraphx_x* migraphx_x_t`) with accessor
  functions, not structs passed or returned by value, and not structs whose
  layout callers can see. Flag a new entry point that takes or returns a struct
  by value, exposes a field directly, returns a pointer into internals, or fixes
  a size or count in the signature where a create/query/free triple would let
  the implementation change later. Name the opaque-handle form it should use
  instead, matching the surrounding entry points.
- Check that the C++ header's inline wrappers stay in step with the C entry
  points they wrap, and that additions are reflected in the Python bindings and
  covered under `test/api`.

### Angle F — IR and extension-contract auditor *(when the IR surface changed, and always at xhigh/max)*

Read the relevant "Extension Patterns" section of `AGENTS.md` — *Adding an
Operation*, *Adding an Optimization Pass*, *Adding a Backend Target*, and the
type-erasure notes — and check the diff against the contracts stated there,
quoting the rule you are checking. Report each violation as a finding: an
operation missing a piece of its required interface or registration, a pass that
is not idempotent or not deterministic, a `compute_shape` that ignores dynamic
shapes, an interface change without regenerated boilerplate. Beyond what the
document states, check that `compute_shape` and `compute` agree on type, lengths
and strides; that a new member is reflected so printing, hashing and
serialization see it and previously serialized programs still load; and that a
matcher's `apply` does not invalidate the instruction it matched.

### Angle G — quality checks (delegated to `/migraphx-simplify`)

Read `.claude/skills/migraphx-simplify/SKILL.md` and run **its Phase 1 review
angles** — reuse, simplification, efficiency, altitude, safety, comments — over
this diff. Use that skill's checklists as written; do not restate them here and
do not re-derive your own version. These findings are reported like any other,
and `--fix` applies them along with the rest.

Then expand past it. That skill is tuned to local cleanups; this review wants
the larger defect those cleanups hint at. For each thing you would have flagged
as a nit, ask what it implies at a larger scale and report *that* instead when
it is real:

- duplicated logic that indicates a missing shared abstraction, or a subsystem
  that should be unified rather than one copy-pasted block edited;
- a special case added to shared infrastructure where the underlying mechanism
  should have been generalized — and what the generalization is;
- an abstraction boundary in the wrong place: state or knowledge leaking across
  a layer, a helper that needs its caller's internals, a pass doing work that
  belongs in lowering or vice versa;
- a design that works for the shapes or types in the diff but will need rework
  for the next one (dynamic shapes, a new dtype, a new arch);
- a change that treats a symptom while the root cause stays.

Drop pure nits: if `/migraphx-simplify` would fix it in one edit and nothing
larger follows from it, it is not worth a finding here. Tag every finding from
this angle with `category: quality` so they can be ranked and applied as a group.

### Angle H — conventions

Read the convention sources that govern the changed code and check the diff
against them:

1. `AGENTS.md` at the repo root — the tracked, canonical version.
2. `CLAUDE.md` at the repo root, and the user-level `~/.claude/CLAUDE.md`.
3. Any `AGENTS.md`, `CLAUDE.md`, or `CLAUDE.local.md` in a directory that is an
   ancestor of a changed file (a directory's file applies only at or below it).

Only flag a violation when you can quote the exact rule and the exact line that
breaks it — no style preferences, no "spirit of the doc" inferences. Name the
file the rule came from and quote it so the report can cite it.

**When no documented rule covers the situation**, do not invent one and do not
stay silent. Grep two or three of the closest comparable files — another
operation in `src/op/`, another pass, another parser in `src/onnx/`, another
kernel header, another test of the same kind — and compare. Flag the diff only
where it diverges from a pattern those files clearly and consistently follow,
and cite the exemplar as `file:line` so the finding stands on evidence rather
than taste. Incidental differences are not findings.

Convention candidates use the same `file`/`line`/`summary` shape; in
`failure_scenario`, state the concrete cost — which rule is broken, or which
established pattern the code now contradicts — instead of a crash. Correctness,
API/ABI, and IR-contract findings always outrank quality, test-coverage, and
convention findings when the output cap forces a cut.

### Angle I — test coverage

Read `AGENTS.md` § *Testing Guidelines* — directory organization, unit tests,
testing passes, numerical verification, verify tests, and test best practices —
and check what the diff adds against it, quoting the guidance you are applying.
Flag:

- a bug fix with no regression test that would have failed before the fix; name
  the test file it belongs in (mirroring the `src/` layout) and the case;
- a new or changed operation with no shape test and no reference-target test, or
  a new ONNX/TF operator with no parse test and no verify test;
- a new or changed pass with no unit test that builds the expected module and
  asserts equality against it;
- changed kernel or GPU numerics with no `test/verify` case;
- a new C API entry point with no `test/api` coverage, and a new Python binding
  with no `test/py` coverage;
- a test that was added but does not actually exercise the change — it would
  pass with the change reverted, or it asserts nothing about the new behavior;
- edge cases the change newly makes reachable that the added tests skip, from
  the list in `AGENTS.md` § *Test Best Practices* (zero-length dimensions,
  dynamic shapes at extreme min/max, mixed type promotion, broadcasting
  asymmetries, reduction axis ordering);
- a deleted or disabled test with no replacement;
- an added test that ignores the repo's test conventions where that will cause
  real friction — for example a verify test sharing a file with another verify
  class.

Before flagging, Grep the mirrored test file for a case that already covers the
path — do not ask for coverage that exists. Do not flag coverage for docs-only,
comment-only, or pure-rename changes.

In `failure_scenario`, state what would break undetected without the test — the
specific bug it would have caught — not that coverage is low. Tag these
`category: test-coverage`. A missing regression test for a bug this diff fixes
ranks with the correctness findings; other coverage gaps rank alongside quality
and conventions.

### Angle J — review precedent

Distilled from the human review comments on PRs #4891–#5108. None of this is
written down in `AGENTS.md` or `/migraphx-simplify` — it is the unwritten
standard this repo's reviewers apply. Flag only what the diff actually does, and
**cite the precedent PR number** in `failure_scenario` so the author sees an
established expectation rather than a personal preference.

**Configuration and knobs**
- New behavior gated on an environment variable where a pass parameter or a
  field in the target's reflected `backend_options` struct would do (#4911,
  #5053, #5028). Env values are read once and cached for the process.
- A test that sets an environment variable — the value leaks into every later
  test in the same process (#5064, #4911).
- An enable/disable flag that is never set to false anywhere, or left behind
  after the feature became the default (#5036, #5030, #5064).
- A CMake workaround enabled for everyone instead of an opt-in cache flag
  guarded on the compiler or condition that actually needs it (#4920, #4941).

**Pass and matcher discipline**
- A precondition checked inside `apply()` that belongs in `matcher()` — use
  `match::nargs`, a named `MIGRAPHX_PRED_MATCHER` predicate, or a composable
  matcher rather than `match::any()` when the surrounding pattern is known
  (#4891, #5105, #4900).
- A predicate helper that mutates the graph: `can_*` functions must stay
  query-only, with every mutation in `apply()` (#4900).
- A pass that needs to run other passes but takes `module&` instead of
  `module_pass_manager&` (#5066).
- A pass inserted into a pipeline without a justified position, a duplicate DCE
  left behind, or an old path the new pass subsumes left in place (#4904,
  #5030, #5096).
- `eval()` or `can_eval()` called from an optimization pass — it walks and
  evaluates the graph at compile time (#4948).
- A rewrite that increases work (hoisting compute above a slice onto a larger
  tensor) or inserts the same instruction more than once (#5004, #5030, #5038).

**Layering and placement**
- Target-specific concepts leaking into target-agnostic code: GPU exceptions
  caught in `pass_manager`, GPU errors added to `errors.hpp`, GPU-only fields in
  `compile_options` (#5021, #5008).
- A helper in the wrong home — put it in the utility header that owns its
  purpose (`fileutils.hpp`, `stringutils.hpp`, `value.cpp`, `device_name.cpp`)
  and keep ONNX-specific logic in the ONNX module (#5024, #4991, #4957, #4895).
- A pass-internal helper promoted to a public header only so a test can reach
  it; prefer black-box coverage through an existing test, and note that Windows
  then needs an export (#4989).

**Error handling**
- A catch-all handler: catch the specific exception, or log and exit. Never
  swallow a failure to keep an optimization "safe" — a shape that fails to
  propagate should surface (#5096, #5021).
- A raise that isn't `MIGRAPHX_THROW` (#4946).

**Shape, layout, and symbolic idioms**
- Linear-index math in an op's `compute` where the output shape can be
  non-standard — use the multi-index `output(i, j)` form, and `with_lens` so a
  permutation propagates (#5046).
- Code assuming the last argument is the output buffer, or forwarding an input
  layout unconditionally (#5030, #5046).
- An axis used without normalization — check for negative, or use
  `ins->normalized_operator()`; assert the invariant if it should already hold
  (#4891).
- `==` / `!=` on symbolic dimensions instead of `same_value` / `same_symbol`
  (#4977).
- A dynamic-shape `compute_shape` that ignores `intersection()` semantics or
  picks a min bound it cannot justify (#4924, #5015, #5043).
- Attribute combinations left unvalidated in `compute_shape` — empty or
  degenerate `starts`/`ends`/`axes`; a zero-length dimension that should become
  `undefined` at parse time (#5088, #4999).

**Naming**
- A name that doesn't say what the code does or doesn't match the local
  convention: encode the side effect (`block_sync_copy_index_if_n`), match
  existing suffixes (`_n`, not `_limit`), prefix by subsystem
  (`replace_onnx_external_weights`), and prefer a precise verb over a vague one
  (#4893, #4957, #5105, #5049, #5030). Renaming an existing shared function
  needs a stated reason (#4893).

**How the test is written** — Angle I asks whether a test exists; this asks
whether the one that exists is written the way reviewers require.
- A pass test asserting on instruction counts or side effects instead of
  building the expected module and comparing against it (#5030, #5105, #4992,
  #5060).
- A fix tested only by pointing at a customer model — distill a minimal repro
  into the matching test file (#5052, #4919).
- An edge case tested at the wrong layer, e.g. contorting the ONNX parser to
  produce a case that belongs in an op-level test (#4999).
- A test carrying ops the case doesn't need, or several verify classes sharing
  one `.cpp` (#5064, #5060).
- A test that runs extra normalizing passes, or trims/resizes the output, so the
  path under test is masked (#4891, #4893).
- A new kernel or optimization tested only for the type or config it was
  developed against when it claims to support more (#4954, #4893).
- A deleted or disabled test — update it if the IR legitimately changed, never
  remove it to go green (#5052).

**Performance claims and heuristics**
- A perf-motivated change with no measured before/after, or one that drops a
  path still faster for some shapes (#5018, #4954, #4948).
- A tuning constant with no stated origin — say where the number came from and
  whether it generalizes past the model that motivated it (#5040, #5038).
- A cache or dedup key derived from the gfx arch name or another non-content
  property; keys must be content-based (#5039, #4992).
- A lock held across expensive work such as a host-to-device copy (#5039).

**What ships with the change**
- A user-visible change with no `Changelog.md` entry, or one filed under the
  wrong category (#4919, #4939, #5038, #4923); internal-only refactors don't
  need one (#4904).
- A touched file whose copyright year range wasn't updated — CI enforces it
  (#4899, #4965, #4966).
- Behavior or options changed without the matching docs page updated, or docs
  claiming support the code doesn't implement (#5028, #4945, #4965, #4946).
- A generated file edited directly instead of its template under `tools/`
  (#4935).
- A new `NOLINT` or `cppcheck-suppress` where the code should be fixed, or tidy
  flags changed in the build instead of `.clang-tidy` (for example adding a type
  to `AllowedTypes`) (#4911, #4977, #4952).

**Public API additions** — extends Angle E.
- An API taking an internal type where a string the implementation parses would
  do (`sym::expr` behind `sym::parse`) (#4946).
- An API added for behavior meant to become the default, which will then do
  nothing (#4946).
- A public signature changed in place instead of adding a forwarding overload
  (#4977).
- A change that makes users include a different header or link new targets —
  that is breaking; call it out (#4961).

**Build and toolchain**
- A local stub, shim, or version workaround for a problem that belongs upstream
  in the pinned dependency (#4988, #4952).
- Docker or prereq changes that drop existing settings (sanitizer flags),
  hand-edit `PATH` for ROCm-installed tools, or pin an old clang-format instead
  of the ROCm-shipped one (#4952).
- CI logic duplicated per stage instead of living in the shared harness (#4910).

## Phase 2 — Verify (1-vote)

Dedup candidates that point at the same line or mechanism, keeping the one with
the most concrete failure scenario. For each remaining candidate, run **one
verifier** via the `Agent` tool: give it the diff, the relevant file(s), and the
candidate, and have it return exactly one of:

- **CONFIRMED** — can name the inputs, shape, or target that trigger it and the
  wrong output or crash. Quote the line.
- **PLAUSIBLE** — mechanism is real, trigger is uncertain (a shape the pass may
  never see, an arch that may not be built, a timing window). State what would
  confirm it.
- **REFUTED** — factually wrong (the code doesn't say that) or guarded
  elsewhere. Quote the line that proves it.

Keep CONFIRMED and PLAUSIBLE. Drop REFUTED.

At **high** and above, verify recall-biased:

> **PLAUSIBLE by default** — do not refute a candidate for being "speculative"
> or "depends on runtime state" when the state is realistic: a non-standard,
> broadcasted, or dynamic shape reaching code that assumes packed; a rarely
> built arch; a zero-length dimension; an fp16 or fp8 intermediate that
> overflows on a large reduction; a race; an `instruction_ref` used after the
> module was mutated. These are PLAUSIBLE.
>
> **REFUTED** only when constructible from the code: factually wrong (quote the
> actual line); provably impossible because a matcher predicate, a
> `compute_shape` check, or an earlier pass excludes it (cite that guard);
> already handled in this diff; or pure style with no observable effect.

Before confirming or refuting a claim about strides, packing, broadcasting, or
dynamic dimensions, read the relevant part of `src/include/migraphx/shape.hpp`
rather than reasoning from the name. At **xhigh** and **max** this is recall
mode — a single non-REFUTED vote carries the finding; do NOT drop on
uncertainty.

## Phase 3 — Sweep for gaps *(xhigh, max)*

Run **one more finder** as a fresh reviewer who has the verified list. Re-read
the diff and enclosing functions looking ONLY for defects not already listed. Do
not re-derive or re-confirm anything already there — the job is gaps. Check the
diff against every trap listed in `AGENTS.md` § *Common Issues*, quoting the one
you are checking, then focus on what a first pass here tends to miss: an
interface change with no regenerated type-erasure boilerplate; a bug fixed with
no regression test added;
a dtype missing from a supported-type list so fusion silently stops matching; a
pass wired into one target's `get_passes()` but not another's; moved or
extracted code that dropped a guard; setup and teardown asymmetry in tests; a
default flipped in a config or an environment knob.

Surface **up to 8 additional candidates**, each naming a defect not already on
the list, and verify them the same way. If nothing is new, return an empty
sweep — do not pad.

## Output

Call the `ReportFindings` tool once to report this review's results with
`{level, findings}`. `findings` is at most the level's cap, ranked most-severe
first; each entry has `file`, `line`, `summary`, `short_summary` — the claim
compressed to ≤60 characters, no rationale or consequence clause —
`failure_scenario`, `category`, and the `verdict` from Phase 2. Use categories
such as `correctness`, `language-pitfall`, `api-abi`, `ir-contract`,
`conventions`, `test-coverage`, `quality` for everything from Angle G, and
`precedent` for an Angle J finding that fits none of the others. If
more than the cap survive, keep the most severe. If nothing survives
verification, call it with an empty array. Do not also print the findings as
text, and do not create or publish an artifact of the review — the tool call is
the report.

If `ReportFindings` is not available, return findings as a JSON array of at most
the level's cap:

```json
[
  {
    "file": "src/targets/gpu/lowering.cpp",
    "line": 123,
    "summary": "one-sentence statement of the bug",
    "category": "correctness",
    "failure_scenario": "concrete inputs/state → wrong output/crash"
  }
]
```

Ranked most-severe first. If nothing survives verification, return `[]`.

## Running without the Agent tool

If the `Agent` tool isn't available, the multi-agent fan-out and the subagent
verify pass can't run. Work through every angle above yourself, in this same
context, in one pass — including one pass per language present — and do not skip
angles for lack of fan-out. Phase 2 becomes dedup and self-check: dedup
near-duplicates, then re-check each remaining candidate against the diff before
keeping it, dropping anything you can't back with a concrete failure scenario.
State clearly in your summary that this was a single-pass review without the
fan-out, so whoever reads it isn't misled about what actually ran.

## Applying fixes (`--fix`)

Only when the `--fix` flag was passed. After producing the findings list, apply
every class of finding to the working tree — correctness, language-pitfall,
API/ABI, IR-contract, and convention findings, **and the `quality` findings from
Angle G**. For those, apply the fix `/migraphx-simplify` would have made
(that skill's Phase 2 describes how it applies its own findings); running this
review with `--fix` should leave the tree in the state a bug-fix pass followed by
`/migraphx-simplify` would have.

For `test-coverage` findings, add the missing test when it is a small,
clearly-derivable case that follows the patterns already in the mirrored test
file — a regression test for a bug you just fixed is the common case. When the
test needs a new fixture, a golden model, or a judgement call about what to
assert, skip it and say what the test should cover.

Skip any finding whose fix would change intended behavior, require changes well
outside the reviewed diff, or that you judge to be a false positive — note the
skip rather than arguing with it. **Never modify or weaken an existing test to
make a fix pass**; if a fix breaks a test, the fix is wrong. Then call
`ReportFindings` again with the same findings, each carrying an `outcome`:
`fixed`, `no_change_needed` (the finding was wrong or already handled), or
`skipped` (real but not applied). Do not repeat the findings as text; after the
call, give one line per skipped finding saying why. If `ReportFindings` isn't
available, finish with a brief summary of what was fixed and what was skipped,
grouped by category.

Without `--fix`, do not modify any file — the report is the only output.

## Posting to GitHub (`--comment`)

Only when the `--comment` flag was passed. After producing the findings list, if
the review target is a GitHub PR, post each finding as an inline PR comment via
`mcp__github_inline_comment__create_inline_comment` (one call per finding;
include a suggestion block only when it fully fixes the issue). If that tool is
not available in this session, fall back to `gh api`
(repos/{owner}/{repo}/pulls/{pr}/comments) or print the findings instead. If the
target is not a PR, print the findings to the terminal and note that
`--comment` was ignored.

## If findings are fixed later

Whenever reported findings get fixed later in this session — the user asks you
to fix them, or later work fixes them incidentally — you MUST call
`ReportFindings` again with the same findings, each carrying an `outcome`:
`fixed`, `no_change_needed`, or `skipped`. Do not repeat the findings as text.
Make that call immediately after the fixes land, before any prose summary; the
host UI's per-finding status updates only from it, and without it the findings
stay marked unresolved.

## Publishing a shareable review (Artifact)

Only when the review's output contract is *not* the `ReportFindings` tool call
(that contract forbids publishing). Publish the findings so they can be shared
and iterated on outside the terminal:

1. Load the `artifact-design` skill (utilitarian treatment — this is a
   document).
2. Write the findings to an HTML file: one section per finding with the file
   path and line, the one-line summary, the concrete failure scenario, and the
   relevant code snippet. If nothing survived verification, the page says so in
   one line.
3. Call the `Artifact` tool with that file path.
4. End the page body with this line verbatim:

   > Paste this URL back into Claude Code to keep iterating on these findings.

Skip this step if the review was invoked only to feed another tool.

## Out of scope

Do not flag, and treat as false positives:

- Pre-existing issues on lines the diff did not touch, unless the diff
  re-exposes the bug or the touched function is the one that fails.
- Anything the linters already catch — clang-format, and the clang-tidy and
  cppcheck warnings that `make -j<N> analyze` reports (see `AGENTS.md`
  § *Linting* for what is enabled). CI runs those.
- Coverage gaps that Angle I rules out: a path an existing test already
  exercises, or a docs-only, comment-only, or pure-rename change. Coverage that
  the diff genuinely leaves untested is in scope — report it.
- Style preferences that no convention source states and no established pattern
  in comparable files supports.
- Changes in behavior that are plainly the intent of the change.

Do not build the project or run the test suite as part of the review — ROCm
builds are slow and this is a reading task. Report what the code says.

## After the review

After the findings are reported (and applied, when `--fix` was passed): if
`/verify` has NOT run this session and the diff has a runtime surface (not
test-only or docs-only per the pre-ship exemptions), invoke `/verify` now — this
review checks that the diff reads right; `/verify` checks that it runs right.
State which you did.
