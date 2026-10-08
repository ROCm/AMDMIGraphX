---
description: "Code review the changed MIGraphX code for correctness bugs, language-specific pitfalls, C/C++ API-ABI breakage, missing test coverage, and convention violations, with a verify pass that drops false positives. The quality checklist is delegated to /migraphx-simplify rather than repeated. Effort levels low through max; --comment posts inline PR comments, --fix applies every class of finding, including the quality findings that clear Angle G's bar. A bare --fix or --comment after a review already ran this session applies or posts that review's findings instead of reviewing again. --select opens a checkbox picker so only the chosen findings are fixed or posted."
allowed-tools: Bash(git diff:*), Bash(git status:*), Bash(git log:*), Bash(git show:*), Bash(git blame:*), Bash(git rev-parse:*), Bash(git merge-base:*), Bash(git branch:*), Bash(git fetch:*), Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh api:*), Bash(grep:*), Bash(find:*), Read, Grep, Glob, Edit, Write, Agent, Skill, ReportFindings, AskUserQuestion, mcp__github_inline_comment__create_inline_comment, mcp__review-picker__select_findings
---

# migraphx-code-review

Usage: `/migraphx-code-review [low|medium|high|xhigh|max] [--fix] [--comment] [--select] [<target>]`

Pick the effort level from the first argument; if none is given, use the session
effort, defaulting to **medium**. Exception: when this review runs as part of
Copilot code review (GitHub Copilot is the agent performing the review), default
to **xhigh** regardless of the session or model effort level — only an explicit
level argument overrides it. Anything after the level is the review target
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

- **low** — You are doing a **fast sanity pass**, not a review: surface only
  bugs that are obvious from the hunk itself, and say plainly that this was a
  shallow pass so nobody mistakes a clean result for a clean diff.
- **medium** — You are reviewing for **precision**: every finding you surface
  should be one a maintainer would act on.
- **high** — You are reviewing for **recall**: catch every real bug a careful
  reviewer would catch in one sitting. Catching real bugs matters more than
  avoiding false positives. Err on the side of surfacing.
- **xhigh / max** — You are reviewing for **recall** at extra-high (or maximum)
  effort: catch every real bug. Catching real bugs matters more than avoiding
  false positives, because a missed bug ships. Err on the side of surfacing.

## Reusing a completed review

If a review from this skill already completed earlier in this session and the
invocation adds nothing but `--fix` and/or `--comment` — optionally with
`--select` — do **not** review again. Act on the findings that review already
reported: skip Phases 0–3 and *Output*, and go straight to *Choosing which
findings to act on* (when `--select` was passed), then *Applying fixes* and
*Posting to GitHub*, with the existing findings list. Say in one line that you
are applying the findings from the earlier review rather than running a new one.

`--select` on its own is not a reuse trigger, because it acts on nothing by
itself. Treat it the same as no flag at all: run a fresh review, then say the
flag had nothing to act on.

Run a fresh review instead when any of these holds:

- no review from this skill has completed in this session;
- the invocation names an effort level or a `<target>` — that is a request for a
  new review, even alongside `--fix`;
- the diff moved since that review ran, so its findings may no longer point at
  the current lines. Check with one `git status --short` (plus `git diff --stat`
  when a hunk-level check is needed) against the file list the earlier review
  worked from — Phase 0's list, or, after a `low` review, the files its diff
  touched. Your own `--fix` edits from an earlier invocation do not count as
  movement; changes the user made do. When it moved, review again and say why.

---

## `low` effort — one diff pass, no verify, ≤4 findings

Run only the two turns below: skip Phases 0–3 and the *Output* section. The flag
sections still apply — `--fix`, `--comment`, and `--select` behave exactly as
they do at any other level, acting on the findings Turn 2 produced.

### Turn 1 — read

One tool call: read the unified diff against the merge-base with the remote
integration branch — `git diff $(git merge-base HEAD origin/develop)`, or the
fallback base `/migraphx-simplify` Phase 0 resolves when `origin/develop` does
not exist, or the target passed as an argument. Skip test and fixture hunks (`test/`,
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
`short_summary` (≤60 characters), `failure_scenario`, and `category` (at this
level that is `correctness` or `quality`, since nothing else is in scope). If
nothing qualifies, call it with an empty findings array. Do not also print the
findings as text. If `ReportFindings` is unavailable, print one line per finding as
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

- **languages present**: put each changed file in the **first** bucket it
  matches, so no file feeds two specialists:
  1. Python — `*.py` anywhere (`tools/`, `src/py/`, `src/api/migraphx.py`,
     `test/py/`, `examples/`)
  2. Bash — `*.sh`
  3. CMake — `CMakeLists.txt`, `*.cmake`
  4. CI YAML — `.github/**`
  5. HIP / device — `src/targets/gpu/kernels/**`, `src/targets/gpu/device/**`
  6. MLIR-facing C++ — `src/targets/gpu/mlir.cpp`,
     `src/targets/gpu/fuse_mlir.cpp`
  7. ONNX / TF parsers — `src/onnx/**`, `src/tf/**`
  8. C++ host — every remaining `.cpp`/`.hpp`, wherever it lives (`src/**`,
     `test/**`, and the API generator inputs under `tools/api/**`)
- **C/C++ API surface**: the generator inputs `tools/api/migraphx.h`,
  `tools/api/api.cpp` and `src/api/migraphx.py`; the hand-written
  `src/api/include/migraphx/migraphx.hpp`; and the generated
  `src/api/include/migraphx/migraphx.h` and `src/api/api.cpp`.
- **IR surface**: the operations in `src/include/migraphx/op/**` and
  `src/op/builder/**`, passes in `src/**` with their headers under
  `src/include/migraphx/**`, and `src/targets/*/target.cpp`.

If there are no changes at all, stop and report that there is nothing to review.

## Phase 1 — Find candidates

Launch the angles as **independent agents via the `Agent` tool, all in a single
message** so they run concurrently (`subagent_type: general-purpose`; they need
Read and Grep). Give each agent the full diff, the classified file list, and the
one angle it owns. Each angle surfaces up to the level's candidate cap — 6 at
`medium` and `high`, 8 at `xhigh` and `max`. Each candidate has `file`, `line`,
a one-line `summary`, and a concrete `failure_scenario`.

**Core angles always run**: A (diff scan), B (removed behavior), C (cross-file),
D (language pitfalls — one agent per language present), G (quality), H
(conventions), I (test coverage), J (review precedent).
**Conditional angles** run when Phase 0 says their surface was touched: E
(C/C++ API and ABI) and F (IR and extension contracts). At `xhigh` and `max`,
run F regardless.

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
target's `get_passes()`. Leave deleted tests to Angle I.

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
  long-running calls, and buffer-protocol shapes and strides. (A new public C++
  API with no binding at all belongs to Angle E when Phase 0 says the API
  surface was touched; when Angle E is not running, report it here so it is not
  missed.)
- **Bash** — unquoted expansions and word splitting, `[` versus `[[`, unguarded
  `cd`, glob expansion on empty matches, a failure silently swallowed by a
  pipeline where the rest of the script checks its exit codes, plus the
  command-invocation rules stated in `AGENTS.md`/`CLAUDE.md`. Do not flag a
  missing `set -euo pipefail` — no script in this repo uses it, so its absence
  is the local convention.
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

The C API is **generated**: `tools/generate.py` produces
`src/api/include/migraphx/migraphx.h` and `src/api/api.cpp` from
`tools/api/migraphx.h`, `tools/api/api.cpp`, and the API description in
`src/api/migraphx.py` (see `src/api/CMakeLists.txt`), and `make generate`
refreshes the checked-in copies. `src/api/include/migraphx/migraphx.hpp` (C++)
is hand-written. Review the generator inputs and the C++ header as the source of
truth, and read the generated header only to see the resulting surface — a diff
that edits the generated files instead of their inputs is itself a finding.

Check that surface for compatibility and durability:

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
- **Robustness against future changes** — opaque handles
  (`typedef struct migraphx_x* migraphx_x_t`) with accessor functions keep both
  the *data layout* and the *function signatures* free to change later, and the
  C API should hand out one rather than exposing either. Flag both failure
  modes, and name the opaque-handle form to use instead, matching the
  surrounding entry points:
  - *robustness against struct changes* — a new entry point that takes or
    returns a struct by value, exposes a field directly, returns a pointer into
    internals, or fixes a size or count in the signature, where a
    create/query/free triple would let the layout change later.
  - *robustness against parameter changes* — a new entry point whose options are
    spelled out as individual parameters, so adding one later changes the
    signature and breaks existing callers. The established form is an options
    handle created and populated through accessors — `migraphx_compile_options_t`
    and `migraphx_onnx_options_t` have a `_create` plus one `_set_*` per option —
    which absorbs a new option without touching the entry point that consumes it.
- Check that the C++ header's inline wrappers stay in step with the C entry
  points they wrap, and that additions are reflected in the Python bindings.
  Missing `test/api` or `test/py` coverage is Angle I's finding, not this
  angle's — report the surface mismatch here and leave the coverage gap there.

### Angle F — IR and extension-contract auditor *(when the IR surface changed, and always at xhigh/max)*

Read the relevant part of `AGENTS.md` § *Extension Patterns* — *Adding an
Operation*, *Adding an Optimization Pass*, *Adding a Backend Target* — plus
§ *Type Erasure System*, and check the diff against the contracts stated there,
quoting the rule you are checking. (The type-erasure templates are
`tools/include/*.hpp`; `make generate` runs `tools/te.py` over them and writes
`src/include/migraphx/<name>.hpp`.) Report each violation as a finding: an
operation missing a piece of its required interface or registration, a pass that
is not idempotent or not deterministic, a `compute_shape` that ignores dynamic
shapes, an interface change without regenerated boilerplate. Beyond what the
document states, check that `compute_shape` and `compute` agree on type, lengths
and strides; that a new member is reflected so printing, hashing and
serialization see it; and that a matcher's `apply` does not invalidate the
instruction it matched.

**Operators do not maintain backwards compatibility.** Do not flag an operator
whose attributes, semantics, or serialized form changed as a compatibility
break, and never ask for `program_file_version` to be incremented because an
operator changed. That constant lives in `src/include/migraphx/program.hpp` and
gates loading a serialized `.mxr`: `program::from_value` throws on any mismatch,
so a bump invalidates every `.mxr` file in existence. It is incremented **only
when the structure of the IR itself changes** — how programs, modules, and
instructions are laid out in the serialized value — not when an individual
operator changes. Flag it in both directions: an IR structure change with no
bump, where an old `.mxr` is parsed against the new structure instead of being
rejected with the version-mismatch error; and a bump attached to an
operator-only change, which invalidates every saved `.mxr` for nothing.

### Angle G — quality checks (delegated to `/migraphx-simplify`)

Read `.claude/skills/migraphx-simplify/SKILL.md` and run **its Phase 1 review
angles** — reuse, simplification, efficiency, altitude, safety, comments — over
this diff. Use that skill's checklists as written; do not restate them here and
do not re-derive your own version. What that pass produces is the raw material
for this angle, not its output: keep a quality finding only if it clears the bar
set below, and report the ones that do like any other finding, with `--fix`
applying them along with the rest.

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
larger follows from it, it is not worth a finding here. That deliberately
narrows the output to the findings that carry weight — this review is not a
substitute for running `/migraphx-simplify`, which still catches the local
cleanups it drops. Tag every finding from this angle with `category: quality` so
they can be ranked and applied as a group.

### Angle H — conventions

Read the convention sources that govern the changed code and check the diff
against them:

1. `AGENTS.md` at the repo root — the tracked, canonical version.
2. `CLAUDE.md` at the repo root, and the user-level `~/.claude/CLAUDE.md` if it
   exists.
3. Any `AGENTS.md`, `CLAUDE.md`, or `CLAUDE.local.md` in a directory that is an
   ancestor of a changed file (a directory's file applies only at or below it).

Only flag a violation when you can quote the exact rule and the exact line that
breaks it — no style preferences, no "spirit of the doc" inferences. Name the
file the rule came from and quote it so the report can cite it.

The test rules written in `AGENTS.md` are yours, not Angle I's or Angle J's: the
expected-module form for pass tests, one verify class per `.cpp`, and the
numerical-verification guidance. Angle I reports that a test is *missing*; you
report a test that exists but breaks one of those written rules, quoting it.

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
API/ABI, and IR-contract findings outrank quality, test-coverage, and convention
findings when the output cap forces a cut. The one exception is set by Angle I:
a missing regression test for a bug this diff fixes ranks with the correctness
findings, not below them.

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
- edge cases the change newly makes reachable that the added tests skip, from
  the list in `AGENTS.md` § *Test Best Practices* (zero-length dimensions,
  dynamic shapes at extreme min/max, mixed type promotion, broadcasting
  asymmetries, reduction axis ordering);
- a deleted or disabled test — updating one is fine when the IR legitimately
  changed, but never deleting or disabling one to make a change pass.

This angle asks whether the coverage **exists**. Whether an existing test is
*written* the way reviewers require — it doesn't reach the changed path, it
asserts nothing that would fail with the change reverted, it carries ops it
doesn't need, it runs extra passes that mask the path — belongs to Angle J. A
test that breaks a rule stated in `AGENTS.md`, such as a verify test sharing a
file with another verify class, belongs to Angle H so the finding quotes the
written rule.

Before flagging, Grep the mirrored test file for a case that already covers the
path — do not ask for coverage that exists. Do not flag coverage for docs-only,
comment-only, or pure-rename changes.

In `failure_scenario`, state what would break undetected without the test — the
specific bug it would have caught — not that coverage is low. Tag these
`category: test-coverage`. A missing regression test for a bug this diff fixes
ranks with the correctness findings; other coverage gaps rank alongside quality
and conventions.

### Angle J — review precedent

Distilled from ~2,000 human review comments on the last 1000 PRs (#3959–#5108) —
the standard this repo's reviewers apply beyond what `AGENTS.md` and
`/migraphx-simplify` already state. Flag only what the diff actually does, and
**cite the precedent PR number** in `failure_scenario` so the author sees an
established expectation rather than a personal preference. If a point here turns
out to be stated explicitly in an `AGENTS.md` rule, report it from Angle H with
the quote instead, so the finding cites the written rule rather than a PR.

**Configuration and knobs**
- New behavior gated on an environment variable where a pass parameter, a field
  on the `target`/`context`, or an entry in `compile_options::backend_options`
  would do — the target reads that map (optionally deserializing it into a
  reflected struct of its own with `from_value`) instead of calling `getenv`
  (#4626, #4651, #4795, #4831, #4882, #4911, #5028, #5053). Env values are read
  once and cached for the process, which is why they can't be varied per compile;
  read the ones that remain through the memoized `MIGRAPHX_<NAME>{}` objects
  rather than at the use site (#4710).
- A test that sets an environment variable — the value leaks into every later
  test in the same process. Add a parameter to the pass and pass it in, the way
  `test/fuse_attention.cpp` does with
  `run_pass(p1, {.flash_decoding_num_splits = 2})` (#4294, #4911, #5064).
- An enable/disable flag that is never set to false anywhere, or left behind
  after the feature became the default; a flag gating behavior that should
  simply always be on needs a stated reason why it can't be (#4725, #4732,
  #4770, #5030, #5036, #5064).
- A new environment variable where an existing one already covers the case, or
  several boolean knobs where one knob taking a value would do — reviewers asked
  for a single variable listing the types to disable rather than one flag per
  type (#4535, #4580).
- An env variable declared inside a device kernel, which cannot work (#4363).
- A CMake workaround enabled for everyone instead of an opt-in cache flag
  guarded on the compiler or condition that actually needs it (#4920, #4941,
  #4952).

**Pass and matcher discipline**
- A precondition checked inside `apply()` that belongs in `matcher()` — use
  `match::nargs`, a named `MIGRAPHX_PRED_MATCHER` predicate, or a composable
  matcher rather than `match::any()` when the surrounding pattern is known
  (#4677, #4725, #4831, #4841, #4880, #4891, #4900, #5105).
- A predicate helper that mutates the graph: `can_*` functions must stay
  query-only, with every mutation in `apply()`, so an early return can't leave
  half a rewrite applied (#4900). Do the eligibility check before inserting
  anything rather than inserting and then calling `remove_instruction` (#4994).
- A transformation placed in the wrong pass. Reshape/transpose/broadcast
  rewrites belong in `simplify_reshapes`, elementwise algebra in
  `simplify_algebra`, redundant-copy elimination in its own pass — never as a
  side effect of lowering (#4546, #4709, #4723, #5014).
- A new finder or matcher that redoes what an existing one already does — extend
  `find_splits`, `find_nested_shape_transforms`, `find_concat_reshape`, or
  `get_splits` (e.g. with a `partial` flag) instead of adding a parallel path
  (#4723, #4724, #5014, #5024, #5064, #5088).
- A matcher that dispatches on operator-name strings where a structural property
  works — prefer `not input->can_eval()` over enumerating `{"add","mul"}` (#4677,
  #4696). Equally, a matcher restricted more than the transform requires, with
  guards copied from a different op's matcher (#4240, #4246, #4292, #4727,
  #4746, #4785).
- A pass that needs to run other passes but takes `module&` instead of
  `module_pass_manager&` (#5066); a transform that must reach weights but walks
  the module directly instead of running the pass manager, so submodules are
  missed (#4957).
- A pass inserted into a pipeline without a justified position, a duplicate DCE
  left behind, DCE run from inside a pass, or an old path the new pass subsumes
  left in place (#4109, #4904, #5030, #5096). Never run DCE on an
  already-compiled program — it produces junk results (#4957).
- `eval()` or `can_eval()` called from an optimization pass — it walks and
  evaluates the graph at compile time (#4948).
- A rewrite that increases work (hoisting compute above a slice onto a larger
  tensor) or inserts the same instruction more than once (#5004, #5030, #5038).
- A throw from a rewrite path: if the precondition doesn't hold, skip the
  transformation instead, and let a later pass handle the case (#4620, #5105).
- **A new representable state added to an op without auditing every existing
  consumer.** When an attribute gains a form it never had, the guards elsewhere
  that used to imply the old form silently stop holding — check every matcher in
  `simplify_reshapes`, `simplify_algebra`, and `simplify_dyn_ops`, not just the
  ones the new test exercises (#5088).

**Layering and placement**
- Target-specific concepts leaking into target-agnostic code: GPU exceptions
  caught in `pass_manager`, GPU errors added to `errors.hpp`, GPU-only fields in
  `compile_options` (use `backend_options`), `gfx` references in a generic pass,
  a backend fusion op emitted by the ONNX frontend, or a target-specific
  decision made inside quantization (#4292, #4303, #4467, #4992, #5008, #5021).
  A new generic pass belongs under `src/`, not `src/targets/gpu/` (#4109).
- A helper in the wrong home — put it in the utility header that owns its
  purpose (`fileutils.hpp`, `stringutils.hpp`, `value.hpp`/`value.cpp`,
  `functional.hpp`, `instruction_traversal.hpp`, `gpu/device_name.cpp`) with its
  own unit tests, and keep ONNX-specific logic in the ONNX module (#4700, #4895,
  #4957, #4991, #5024, #5048, #5088).
- A pass-internal helper promoted to a public header only so a test can reach
  it; prefer black-box coverage through an existing test, and note that Windows
  then needs an export (#4989). Conversely, a helper class used by one
  translation unit should live in that `.cpp`, not a new header (#4770).
- Operator and class implementations left in a header where they should be in
  the `.cpp` — headers everyone includes must stay light (#4496, #4549, #4619,
  #4831).

**Error handling**
- A catch-all handler: catch the specific exception, or log and exit. Never
  swallow a failure to keep an optimization "safe" — a shape that fails to
  propagate should surface (#4978, #5021, #5096). A `try` that only rethrows
  should be deleted (#4770).
- A raise that isn't `MIGRAPHX_THROW`, including in the Python bindings (#4946).
- An error status from an external C API (HSA/HIP) discarded — propagate it, or
  say in a comment why partial results are acceptable (#4496).
- A stub or unimplemented path that returns a plausible constant instead of
  throwing, which hides the missing implementation behind a wrong answer (#4710).
- `assert` used for a condition that causes a real failure — it does not fire in
  release builds, so throw or size the allocation correctly (#4831).
- An error message that doesn't identify what threw or what value was rejected;
  name the component (the builder, not the ONNX op) and print the offending
  value (#4005, #4054).

**Shape, layout, and symbolic idioms**
- Linear-index math in an op's `compute` where the output shape can be
  non-standard — use the multi-index `output(i, j)` form, and `with_lens` so a
  permutation propagates (#5046). A derived shape built assuming a packed layout
  must recompute strides explicitly (#4409).
- Code assuming the last argument is the output buffer, that argument 1 is
  `starts` and 2 is `ends`, that the scale is the first literal, that a batch
  dimension is named "batch", or that an op is already normalized (#4831, #4850,
  #5030, #5088, #5105).
- An axis used without normalization — check for negative, or use
  `ins->normalized_operator()`; assert the invariant if it should already hold
  (#4891).
- `==` / `!=` on symbolic dimensions where the comparison should ignore variable
  metadata — a `sym::expr` carries constraints and optimals, so use
  `sym::same_symbol` (calling `sym::as_symbol` first when needed) to compare
  structural form (#4924, #4925, #4977). Use `shape::same_lens()` for length
  comparison because it also works for dynamic shapes (#4881), and `ndim()`
  rather than `lens().size()` (#4521, #4591).
- A dynamic-shape `compute_shape` that ignores `intersection()` semantics or
  picks a min bound it cannot justify (#4924, #5015, #5043).
- Attribute combinations left unvalidated in `compute_shape` — empty or
  degenerate `starts`/`ends`/`axes`, a duplicate or misordered mode entry, an
  attribute supplied both by attribute and by input, or a zero-length dimension
  that should become `undefined` at parse time (#4290, #4881, #4999, #5088).
- `std::get<int64_t>` on an attribute that may now be symbolic — it throws
  `std::bad_variant_access` out of the middle of a pass with no op name or
  instruction context, since nothing catches around the pipeline (#5088). Read
  attributes with `ins->get_operator().to_value()` rather than `any_cast` to the
  concrete op type (#4725, #5088).
- A literal or limit created as `float` instead of the input's element type,
  which silently promotes an fp16 model (#4067, #4103, #4190, #4518).

**Signatures and parameters**
- `instruction_ref` passed by reference — it is a cheap handle and goes by value
  (#4204); other non-trivial parameters go by `const&` (#4001, #4790).
- An output parameter where a return value works, including "return the index,
  don't take an output parameter" and functions that should return a struct
  rather than write through several references (#3989, #4001, #4095, #4481,
  #4766).
- A sentinel or size-derived presence check where `std::optional` says it
  directly — `attn_bias.has_value()` rather than inspecting `args.size()` or the
  operator name (#4095, #4637, #4703, #4880).
- A new overload where a defaulted parameter would do, including test helpers
  that then read as `run_pass(p1, {.flash_decoding_num_splits = 2})` (#4384,
  #4393, #4626, #4823).
- A parameter, member, or attribute fully derivable from something already
  passed — `sizes.size()` rather than a separate `num_segments`, `s0.max_lens()`
  rather than an `output_lens` attribute, a capturing lambda rather than a
  `user_data` pair (#4290, #4409, #4483, #4496, #4527).

**State, lifetime, and thread safety**
- A `const` method that mutates, whether through a `mutable` member or through
  the impl pointer; hold the state behind a `shared_ptr` initialized in the
  constructor or in `finalize` instead (#4101, #4204, #4549). Cppcheck catches
  the bare `mutable` member, so report the design problem it points at — a
  lazily built cache that is never destroyed, or a getter that computes and
  stores — rather than the keyword.
- A namespace-scope object with a global constructor; wrap it in a function with
  a `static` local and return a reference (#4015, #4037, #4109, #4111, #4197,
  #4469).
- State stashed in `module`, or a `module` mutated from more than one thread —
  the class is not thread-safe (#4626).
- A pointer into an internally allocated or scratch buffer returned to the
  caller: it dangles once the program is destroyed (#4880).
- Iterators taken from two different temporaries, or from a function returning
  by value — copy to a named local first (#4567).

**Logging and diagnostics**
- Status or progress written to `std::cout`; route it through the logger so it
  can be filtered by level, and reserve stdout for intended output such as
  program results, times, and perf reports (#4804, #4861, #4992, #5064).
- A `debug_print` or dump helper routed through the logger or gated on an env
  variable — those exist to be called from a debugger, so they print to
  `std::cout` directly (#4732).
- A warning that fires on a common legitimate case, such as an internal type
  conversion or a value the user already configured (#3985, #4850).
- A failure path silenced rather than demoted — keep the message and lower its
  level (#4861).

**Naming**
- A name that doesn't say what the code does or doesn't match the local
  convention: encode the side effect (reviewers renamed a kernel helper to
  `block_sync_copy_index_if_n` because it calls `__syncthreads`), match existing
  suffixes (`_n`, not `_limit`), prefix an ONNX-only helper with its subsystem,
  read predicates as predicates (`has_*`), state the relation in a threshold
  (`min_partition_threshold`), and prefer a precise verb over a vague one
  (#4893, #4957, #5030, #5049, #5105).
- A name that stops being true when scope widens — reviewers rejected a
  `lower_hip_ops` pass name once it also handled `gpu::contiguous` (#5030); a
  name that collides with an established meaning in this codebase (`ctx`,
  `half`, `time`); and cryptic abbreviations in user-facing output (`[w]` for
  `[warn]`) (#4194, #4384, #4469, #4810).
- Renaming an existing shared function needs a stated reason (#4893).

**How the test is written** — Angle I asks whether a test exists; this asks
whether the one that exists is written the way reviewers require.
- A fix tested only by pointing at a customer model — distill a minimal repro
  into the matching test file (#4919, #5052).
- An edge case tested at the wrong layer, e.g. contorting the ONNX parser to
  produce a case that belongs in an op-level test, or a verify test where
  `test/ref/<op>.cpp` with gold values fits because the input is a literal
  (#4999, #5014, #5068).
- A test carrying ops the case doesn't need, or a redundant trailing
  `add_return` — the last instruction is already the output (#5007, #5060,
  #5064).
- A test that runs extra normalizing or fusion passes, or trims/resizes the
  output, so the path under test is masked; write the replacement instructions
  directly instead of producing them with `fuse_pointwise` (#4626, #4891, #4893,
  #5064).
- A test that does not actually pin the change — either it never reaches the
  changed code because its ops are rewritten away before the new matcher sees
  them, or it reaches the code but asserts nothing that would fail with the
  change reverted (#4176, #4388).
- A test gated on an environment variable or wrapped in `try`/`catch`; check
  every precondition explicitly so the test always runs (#4294). Test against
  `migraphx::module` rather than `migraphx::program` when the pass takes a
  module (#4294).
- An assertion derived by calling the code under test — state expected shapes
  literally in `op_shape_test` rather than computing them with `compute_shape()`
  (#4699). Asserting by instruction count or printed text instead of building
  the expected module breaks a written `AGENTS.md` rule, so Angle H owns that
  one (#4992, #5030, #5060, #5105).
- A "dynamic shape" test whose inputs are all static (#4704).
- A new kernel, mode, or optimization tested only for the type or config it was
  developed against when it claims to support more; a new compile mode should
  run the existing suite under both configurations rather than adding one-off
  tests (#4893, #4954, #4770).
- Gold/expected numeric data with no comment saying how it was produced, and
  ONNX test assets not generated through `test/onnx/gen_onnx.py` (#4041, #4067,
  #4521, #4673).
- An ONNX operator change with only a verify test — parse tests are white-box
  and should build the expected program by hand (#4067, #4093).

The expected-module form for pass tests and one verify class per `.cpp` are
written rules in `AGENTS.md`; Angle H owns those, and a deleted or disabled test
belongs to Angle I.

**ONNX and spec conformance**
- Behavior branched on the opset version where checking whether the input or
  attribute is actually present says the same thing more directly (#4518).
- An optional input accepted but never consumed downstream, or accepted without
  validating the inputs the spec requires alongside it (#4637, #4703).
- A spec assumption that doesn't hold: scale and zero-point may be N-D rather
  than scalar, negative axes need `tune_axis`, an op with several spec outputs
  must return `std::vector<instruction_ref>`, and a zero-element output can be
  legal (#4521, #4571, #4591).
- Type-promotion logic invented for the parser instead of matching the C++
  rules, especially special cases for literals — exceptions make the promotion
  impossible to reason about, and the dynamic branch must behave like the static
  one (#4826).
- A parser limitation worked around by changing an existing operator's
  semantics, where composing existing ops would do (#4880).

**Performance claims and heuristics**
- A perf-motivated change with no measured before/after across models and sizes,
  or one that drops a path still faster for some shapes (#4948, #4954, #5014,
  #5018).
- A tuning constant with no stated origin — say where the number came from and
  whether it generalizes past the model that motivated it; launch-geometry
  constants in particular need measurements across hardware (#4591, #4595,
  #4709, #5038, #5040).
- A cache or dedup key derived from the gfx arch name or another non-content
  property — multiple GPUs share a gfx name; keys must be content-based, and a
  bare hash without the shape (or without collision handling) silently loses
  data (#4992, #5039).
- A lock held across expensive work such as a host-to-device copy, or a new
  in-process cache with a lock where `problem_cache` or ccache already applies —
  it bottlenecks parallel compilation and doesn't persist (#4708, #5039).
- Work proportional to the whole module inside a matcher or finder: traversing
  from `begin()`, or building a map sized to the module, is slow on large models
  and runs on every match (#4152, #4216, #4626, #4727).
- Compile-time cost treated as free — nested `visit`/`visit_all` in lowering, a
  `static_assert` added for an invariant, and a second lookup table all slow
  compilation measurably (#4255, #4591, #4631, #4720).

**Reference and GPU parity**
- A spec-mandated error handled differently on the two paths: the reference
  implementation should throw and the GPU should `MIGRAPHX_ASSERT` and write a
  defined value, rather than reading out of bounds (#4363).
- A GPU path that leaves the tail of a fixed-size output uninitialized where the
  reference zero-fills it — a consumer reading the full output then gathers
  garbage (#4893).
- A `static_assert` or capacity limit added to a JIT kernel without extending
  the lowering fallback to every static case the reference op accepts; a
  `static_assert` is a hard build failure, not a fallback (#4893).
- An allocation hidden inside another operator's implementation — memory
  coloring cannot see it (#4591) — or a host/device copy moved inside a code
  object op, which breaks hipGraph (#5032).
- Launch bounds or occupancy hints set on JIT kernels, which costs the
  optimizations the default range enables (#4217).

**What ships with the change**
- A user-visible change with no `CHANGELOG.md` entry, or one filed under the
  wrong category (#4919, #4923, #4939, #5038). Write it as a complete
  past-tense sentence naming the exact API or operator affected, reference the
  PR number rather than the issue, and merge it into a related existing entry
  instead of duplicating; internal-only refactors and incomplete or experimental
  features don't get one at all (#4373, #4497, #4512, #4567, #4904, #5007,
  #5065).
- A touched file whose copyright year range wasn't updated — CI enforces it
  (#4899, #4965, #4966, #5088).
- Behavior or options changed without the matching docs page updated, or docs
  claiming support the code doesn't implement (#4945, #4946, #4965, #5028).
- A generated file edited directly instead of its template under `tools/` —
  including the type-erasure headers in `src/include/migraphx/`, which come from
  `tools/include/` via `make generate` (#4709, #4935).
- A new `NOLINT` or `cppcheck-suppress` where the code should be fixed, tidy
  flags changed in the build instead of `.clang-tidy` (for example adding a
  cheap-copy type to `AllowedTypes`), or a warning disabled globally to silence
  one site (#4143, #4190, #4801, #4911, #4952, #4977).
- **Unrelated changes bundled in**, at PR granularity — `/migraphx-simplify`
  already flags an unrequested refactor for separation, so what this adds is
  where the split falls: drive-by formatting, `.gitignore` edits, and a second
  feature each belong in their own PR, and a large refactor should land first as
  a no-functional-change PR so the behavior change on top of it is reviewable
  (#4363, #4626, #4663, #4725, #4760, #4803, #4911, #4952).
- A new `TODO` that isn't tracked — file an issue and reference it, or say
  concretely what remains (#4174, #4246, #4875, #4893, #4894).

**Public API additions** — extends Angle E.
- An API taking an internal type where a string the implementation parses would
  do (`sym::expr` behind `sym::parse`) (#4946).
- An API added for behavior meant to become the default, which will then do
  nothing, or a compile option that duplicates an existing environment variable
  (#4893, #4946).
- A public signature changed in place instead of adding a forwarding overload —
  other in-flight PRs call it (#4931, #4977).
- A C++ container passed or returned across the C API boundary; materialize it
  on the wrapper side, since C strings are ABI-safe and `std::vector` is not
  (#4341). A C callback should hand back an opaque extensible handle rather than
  raw strings, registered through the generator's `api.add_callback` (#4780).
- A new parallel entry point where an existing options struct could take one
  more defaulted field — combinations of overloads multiply fast (#4701, #4770,
  #4780, #4823).
- A new enum or constant added to a public C++ header but not mirrored into the
  C API with the conversion overloads that carry it across the boundary — the
  established form is the `to_shape_type` pair in `tools/api/api.cpp`, one
  overload each way (#4770).
- A public API extended ahead of a design that covers the known cases — expose
  the minimal accessor instead (#4803).
- A change that makes users include a different header or link new targets —
  that is breaking; call it out (#4961).

**Build and toolchain**
- A local stub, shim, or version workaround for a problem that belongs upstream
  in the pinned dependency, in rocm-cmake, or in the compiler (#4765, #4952,
  #4988). A release-driven hack may go to a release branch but not to `develop`
  (#4765).
- Docker or prereq changes that drop existing settings (sanitizer flags),
  hand-edit `PATH` for ROCm-installed tools, pin an old clang-format instead of
  the ROCm-shipped one, install a toolchain with `curl | bash`, or base a CI
  image on a large uncontrolled upstream image (#4466, #4623, #4714, #4952).
- CI logic duplicated per stage instead of living in the shared harness, or
  behavior that depends on stage names (#4682, #4910).
- CMake that links raw library paths instead of imported targets, installs an
  embedded artifact that should be an object library folded into the static
  library, makes each consumer repeat requirements that belong on the
  `INTERFACE`, adds an unprefixed global or cache variable, or conditions a core
  dependency on a target-specific one (#3992, #4243, #4345, #4714, #4765, #4791,
  #4839).

## Phase 2 — Verify (1-vote)

Dedup candidates that point at the same line **and** the same mechanism, keeping
the one with the most concrete failure scenario. Two candidates on one line that
describe different mechanisms are two findings — Phase 1 recorded both on
purpose, so do not collapse them. For each remaining candidate, run **one
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
rather than reasoning from the name.

Every level that reaches Phase 2 runs exactly one verifier per candidate —
`medium` and above; `low` has no verify step at all. At **high** and above, any
candidate the verifier does not REFUTE survives to the report: uncertainty is
not a reason to drop it, because that is what the PLAUSIBLE verdict is for and
the reader sees the verdict. At **medium** the precision stance governs instead
— keep a PLAUSIBLE candidate only when a maintainer would act on it.

## Phase 3 — Sweep for gaps *(xhigh, max)*

Run **one more finder** as a fresh reviewer who has the verified list. Re-read
the diff and enclosing functions looking ONLY for defects not already listed. Do
not re-derive or re-confirm anything already there — the job is gaps. Check the
diff against every trap listed in `AGENTS.md` § *Common Issues*, quoting the one
you are checking, then focus on what a first pass here tends to miss: an
interface change with no regenerated type-erasure boilerplate;
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
`failure_scenario`, `category`, and the `verdict` from Phase 2.

Pick `category` by which angle produced the finding: `correctness` (A, B, C),
`language-pitfall` (D), `api-abi` (E), `ir-contract` (F), `quality` (G),
`conventions` (H), `test-coverage` (I), and `precedent` (J). When an Angle J
finding fits one of the earlier categories better — a symbolic-comparison bug is
`correctness` — use that one and keep the PR citation in `failure_scenario`.

If more than the cap survive, keep the most severe: correctness, API/ABI, and
IR-contract findings outrank quality, test-coverage, and convention findings,
with one exception — a missing regression test for a bug this diff fixes ranks
with the correctness findings. If nothing survives
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
    "short_summary": "claim in ≤60 characters",
    "category": "correctness",
    "verdict": "CONFIRMED",
    "failure_scenario": "concrete inputs/state → wrong output/crash"
  }
]
```

Ranked most-severe first. If nothing survives verification, return `[]`.

## Running without the Agent tool

If the `Agent` tool isn't available, the multi-agent fan-out and the subagent
verify pass can't run. Work through every angle above yourself, in this same
context, in one pass — including one pass per language present — and do not skip
angles for lack of fan-out. Phase 2 becomes dedup and self-check: dedup only
candidates that share both a line and a mechanism, then re-check each remaining
candidate against the diff and assign it the same CONFIRMED / PLAUSIBLE /
REFUTED verdict a verifier would — drop what you can REFUTE against the code,
not what you are merely unsure of.
State clearly in your summary that this was a single-pass review without the
fan-out, so whoever reads it isn't misled about what actually ran.

## Choosing which findings to act on (`--select`)

Only when the `--select` flag was passed. It filters what `--fix` and
`--comment` act on; on its own it changes nothing, so if neither of those was
also passed, say the flag had nothing to act on and stop after the report.

Run this once the findings are visible to the user and **before** applying or
posting anything — normally right after this invocation's `ReportFindings` call,
or, on the reuse path where no new report is produced, after restating the
earlier review's findings in rank order so the user has something to pick from.
Offer every reported finding, in that same ranked order, using the first of
these that works:

1. `mcp__review-picker__select_findings` — a checkbox dialog. Pass one item per
   finding with `id` set to its 1-based rank and `label` set to
   `file:line — short_summary`, plus a `message` naming the action ("Which
   findings should I fix?" / "…post as PR comments?"). It returns `action` and
   the selected ids.
2. `AskUserQuestion` with `multiSelect: true`, when the picker is unavailable or
   returns `action: "unsupported"`. One option per finding, `short_summary` as
   the label and `failure_scenario` as the description, split across as many
   questions as it takes — four options each, four questions per call.
3. Printing the numbered findings and asking which to act on, when neither tool
   is available.

Act only on the selected findings. Treat `decline`, `cancel`, or an empty
selection as "act on nothing" — never fall back to applying everything, because
the user asked to choose. In a headless run where no picker can be answered, do
the same and say the selection could not be made.

Every reported finding still appears in the follow-up `ReportFindings` call —
the one `--fix` always makes, and on a `--select --comment` run with no `--fix`,
a call you make after posting so the unselected findings do not sit unresolved
in the UI. The ones the user did not select carry `outcome: skipped`; say they
were skipped as unselected, and do not argue for them.

The checkbox dialog comes from a small MCP server kept with this skill at
`.claude/mcp/select_findings.mjs`. It is not registered by default — enable it
once per machine from the repo root with
`claude mcp add review-picker -- node "$PWD/.claude/mcp/select_findings.mjs"`
and restart the session. Without it, `--select` falls through to
`AskUserQuestion`, so the flag works either way.

## Applying fixes (`--fix`)

Only when the `--fix` flag was passed. The findings list is either the one this
invocation just produced, or — per *Reusing a completed review* — the one an
earlier review in this session reported; on the reuse path you arrive here
directly, without re-reviewing and after the selection step when `--select` was
passed. When `--select` was also passed, act only on the findings chosen
there. Apply every finding to the working tree, whatever its `category` —
including `category: quality`. For a quality finding, apply the fix
`/migraphx-simplify` would have made (that skill's Phase 2 describes how it
applies its own findings).

Either way the tree ends up short of a full `/migraphx-simplify` run: at
`medium` and above because Angle G reports only the quality findings that clear
its bar, and at `low` because no quality review ran at all. Say which case
applies and suggest running `/migraphx-simplify` for the local cleanups.

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

Only when the `--comment` flag was passed. The findings list is either the one
this invocation just produced, or — per *Reusing a completed review* — the one an
earlier review in this session reported; on the reuse path you arrive here
directly, without re-reviewing and after the selection step when `--select` was
passed, and post against the PR that review targeted. When `--select` was
also passed, post only the findings chosen there. If the review target is a
GitHub PR, post each finding as an inline PR comment via
`mcp__github_inline_comment__create_inline_comment` (one call per finding;
include a suggestion block only when it fully fixes the issue). If that tool is
not available in this session, fall back to `gh api`
(repos/{owner}/{repo}/pulls/{pr}/comments) or print the findings instead. If the
target is not a PR, print the findings to the terminal and note that
`--comment` was ignored.

Begin every comment body with `[agent]: ` so a reader can tell it was generated
by an agent rather than written by the account posting it. This applies to both
paths above and to any summary or review-level comment, not just the inline
ones. Keep the marker outside a `suggestion` block — a suggestion's contents are
committed verbatim when someone accepts it, so a marker inside one lands in the
source.

### Writing the comment

Keep every comment clear and concise. The body is normally one short paragraph:
the defect in a single sentence, then the concrete failure it causes — the
input, shape, type, or target that triggers it and what goes wrong. Aim for
under 80 words, and when a finding needs more, spend them on the failure
scenario.

- Lead with the defect. No preamble, no restating what the diff does, no praise,
  no summary of the surrounding code.
- Default to full sentences with the technical terms spelled out. Use a short
  bullet list, a small table, or a fenced `mermaid` diagram (GitHub renders
  them) when it explains the problem *more* concisely than prose — an ordering
  or lifetime that goes wrong across several steps, a shape or layout
  transformation, a handful of enumerable cases or shapes. Skip it when prose is
  just as short, and keep it small: a few bullets or nodes, not a document. No
  labels the reader has to cross-reference back to.
- Do not quote the lines being commented on — the comment is already anchored to
  them. Quote a different line only when the failure depends on it.
- Give the fix as a short clause, or as a `suggestion` block when it fully fixes
  the issue. Not both, and do not offer alternatives to choose between.
- Cite the source in a brief parenthetical when the finding rests on one — the
  `AGENTS.md` rule or the precedent PR number — rather than a separate paragraph.
- State what is wrong, not how sure you are: drop "consider", "you may want to",
  and "it seems". For a `PLAUSIBLE` verdict, name the condition that triggers it
  instead of hedging the claim.

## If findings are fixed later

Whenever reported findings get fixed later in this session — the user asks you
to fix them, or later work fixes them incidentally — you MUST call
`ReportFindings` again with the same findings, each carrying an `outcome`:
`fixed`, `no_change_needed`, or `skipped`. Do not repeat the findings as text.
Make that call immediately after the fixes land, before any prose summary; the
host UI's per-finding status updates only from it, and without it the findings
stay marked unresolved.

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

The report — plus the applied edits when `--fix` was passed — is the whole
deliverable. This review checks that the diff *reads* right; nothing here
checks that it *runs* right, so when the diff has a runtime surface, close by
naming in one line what still needs to be built and tested before it ships.
Do not run those builds or tests yourself; see *Out of scope*.
