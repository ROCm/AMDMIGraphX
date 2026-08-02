---
description: "Code review the current diff for correctness bugs and reuse/simplification/efficiency cleanups at a given effort level (low/medium — fewer, high-confidence findings; high through max — broader coverage, may include uncertain findings). Pass --comment to post findings as inline PR comments, or --fix to apply the findings to the working tree after the review."
allowed-tools: Bash(git diff:*), Bash(git status:*), Bash(git log:*), Bash(git show:*), Bash(git blame:*), Bash(git rev-parse:*), Bash(git merge-base:*), Bash(git branch:*), Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh api:*), Bash(grep:*), Bash(find:*), Read, Grep, Glob, Edit, Agent, ReportFindings, Artifact, Skill, mcp__github_inline_comment__create_inline_comment
---

# Code review

Usage: `/migraphx-code-review [low|medium|high|xhigh|max] [--fix] [--comment] [<target>]`

Pick the effort level from the first argument; if none is given, use the session
effort, defaulting to **medium**. Anything after the level is the review target
(a PR number, branch name, ref range, or file path). Run the pipeline for that
level and nothing more.

| Level | Pipeline | Cap |
|-------|----------|-----|
| `low` | 1 diff pass → no verify | 4 findings |
| `medium` | 3+5 angles × 6 candidates → 1-vote verify | 8 findings |
| `high` | 3+5 angles × 6 candidates → 1-vote verify (recall-biased) | 10 findings |
| `xhigh` | 5+5 angles × 8 candidates → 1-vote verify → sweep | 15 findings |
| `max` | 5+5 angles × 8 candidates → 1-vote verify → sweep | 15 findings |

State the level's tagline at the top of the review, then open with the stance
for that level:

- **medium** — You are reviewing for **precision** at medium effort: every
  finding you surface should be one a maintainer would act on.
- **high** — You are reviewing for **recall** at high effort: catch every real
  bug a careful reviewer would catch in one sitting. At this level, catching
  real bugs matters more than avoiding false positives. Err on the side of
  surfacing.
- **xhigh / max** — You are reviewing for **recall** at extra-high (or maximum)
  effort: catch every real bug. At this level, catching real bugs matters more
  than avoiding false positives — a missed bug ships. Err on the side of
  surfacing.

---

## `low` effort — one diff pass, no verify, ≤4 findings

At this level, run only the two turns below; skip every phase that follows.

### Turn 1 — read

One tool call: read the unified diff (`git diff @{upstream}...HEAD; git diff HEAD`
to cover both committed and uncommitted changes, or `git diff main...HEAD` /
the target passed as an argument). Skip test/fixture
hunks (`test/`, `spec/`, `__tests__/`, `*_test.*`, `*.test.*`,
`fixtures/`, `testdata/`) — test-file changes are not reviewed at this level.
No subagents, no full-file reads.

### Turn 2 — findings

Flag runtime-correctness bugs visible from the hunk alone: inverted/wrong
condition, off-by-one, null/undefined deref where adjacent lines show the value
can be absent, removed guard, falsy-zero check, missing `await`,
wrong-variable copy-paste, error swallowed in a catch that should propagate.
Also flag — still from the hunk alone — new code that duplicates an existing
helper visible in the diff context, and dead code the diff leaves behind.

Do **not** flag style, naming, perf, missing tests, or anything outside the
hunk.

Report at most **4 findings**, most-severe first, in one
`ReportFindings` call with `{level, findings}` — each entry has
`file`, `line`, `summary`, `short_summary` (≤60 characters), and
`failure_scenario`. If nothing qualifies, call it with an empty findings
array. Do not also print the findings as text.

If the `ReportFindings` tool is not available: output at most **4 findings**,
most-severe first, one line each:
`path/to/file.ext:123 — what's wrong and the concrete failure`. If nothing
qualifies, output exactly `(none)`.

---

## Phase 0 — Gather the diff

Run `git diff @{upstream}...HEAD` (or `git diff main...HEAD` / `git diff HEAD~1`
if there's no upstream) to get the unified diff under review. If there are
uncommitted changes, or the range diff is empty, also run `git diff HEAD` and
include the working-tree changes in scope — the review often runs before the
commit. If a PR number, branch name, or file path was passed as an argument,
review that target instead. Treat this diff as the review scope.

## Phase 1 — Find candidates

At **medium** and **high**: 3 correctness angles + 3 cleanup angles + 1 altitude
angle + 1 conventions angle. Run **8 independent finder angles** (A, B, C, then
Reuse, Simplification, Efficiency, Altitude, Conventions) via the `Agent` tool.
Each surfaces **up to 6 candidate findings** with `file`, `line`, a one-line
`summary`, and a concrete `failure_scenario`.

At **xhigh** and **max**: 5 correctness angles + 3 cleanup angles + 1 altitude
angle + 1 conventions angle. Run **10 independent finder angles** (A through E,
then the same cleanup, altitude, and conventions angles) via the `Agent` tool.
Each surfaces **up to 8 candidate findings**. Do NOT let one angle's conclusions
suppress another's — if two angles flag the same line for different reasons,
record both.

If the `Agent` tool is not available in your current tool set, do not error —
perform each angle (and each verification) yourself, sequentially, in this
context; see *Running without the Agent tool* below.

### Angle A — line-by-line diff scan

Read every hunk in the diff, line by line. Then Read the enclosing function for
each hunk — bugs in unchanged lines of a touched function are in scope (the PR
re-exposes or fails to fix them). For every line ask: what input, state, timing,
or platform makes this line wrong? Look for inverted/wrong conditions,
off-by-one, null/undefined deref, missing `await`, falsy-zero checks,
wrong-variable copy-paste, error swallowed in catch, unescaped regex metachars.

### Angle B — removed-behavior auditor

For every line the diff DELETES or replaces, name the invariant or behavior it
enforced, then search the new code for where that invariant is re-established.
If you can't find it, that's a candidate: a removed guard, a dropped error
path, a narrowed validation, a deleted test that was covering a real case.

### Angle C — cross-file tracer

For each function the diff changes, find its callers (Grep for the symbol) and
check whether the change breaks any call site: a new precondition, a changed
return shape, a new exception, a timing/ordering dependency. Also check callees:
does a parallel change in the same PR make a call unsafe?

### Angle D — language-pitfall specialist *(xhigh, max)*

Scan for the classic pitfalls of the diff's language/framework — for example:
JS falsy-zero, `==` coercion, closure-captured loop var; Python mutable default
args, late-binding closures; Go nil-map write, range-var capture; SQL injection;
timezone/DST drift; float equality. Flag any instance the diff introduces.

### Angle E — wrapper/proxy correctness *(xhigh, max)*

When the PR adds or modifies a type that wraps another (cache, proxy, decorator,
adapter): check that every method routes to the wrapped instance and not back
through a registry/session/global — e.g. a caching provider holding a
`delegate` field that resolves IDs via `session.get(...)` instead of
`delegate.get(...)` will re-enter the cache or recurse. Also check that the
wrapper forwards all the methods the callers actually use.

### Reuse

The angles above hunt for bugs; this one and the next two hunt for cleanup in
the changed code. Flag new code that re-implements something the codebase
already has — Grep shared/utility modules and files adjacent to the change,
and name the existing helper to call instead.

### Simplification

Flag unnecessary complexity the diff adds: redundant or derivable state,
copy-paste with slight variation, deep nesting, dead code left behind. Name
the simpler form that does the same job.

### Efficiency

Flag wasted work the diff introduces: redundant computation or repeated I/O,
independent operations run sequentially, blocking work added to startup or
hot paths. Also flag long-lived objects built from closures or captured
environments — they keep the entire enclosing scope alive for the object's
lifetime (a memory leak when that scope holds large values); prefer a
class/struct that copies only the fields it needs. Name the cheaper
alternative.

### Altitude

Check that each change is implemented at the right depth, not as a fragile
bandaid. Special cases layered on shared infrastructure are a sign the fix
isn't deep enough — prefer generalizing the underlying mechanism over adding
special cases.

### Conventions (CLAUDE.md)

Find the CLAUDE.md files that govern the changed code: the user-level
~/.claude/CLAUDE.md, the repo-root CLAUDE.md, plus any CLAUDE.md or
CLAUDE.local.md in a directory that is an ancestor of a changed file (a
directory's CLAUDE.md only applies to files at or below it). Read each one
that exists, then check the diff for clear violations of the rules they state.

Only flag a violation when you can quote the exact rule and the exact line
that breaks it — no style preferences, no vague "spirit of the doc"
inferences. In the finding, name the CLAUDE.md path and quote the rule so the
report can cite it. If no CLAUDE.md applies, return nothing for this angle.

Cleanup, altitude, and conventions candidates use the same
`file`/`line`/`summary` shape; in `failure_scenario`, state the concrete
cost (what is duplicated, wasted, harder to maintain, or which CLAUDE.md rule
is broken) instead of a crash. Correctness bugs always outrank cleanup,
altitude, and conventions findings when the output cap forces a cut.

Pass every candidate with a nameable failure scenario through — finders that
silently drop half-believed candidates bypass the verify step and are the
dominant cause of misses.

## Phase 2 — Verify (1-vote)

Dedup candidates that point at the same line/mechanism, keeping the one with
the most concrete failure scenario (at high effort: dedup near-duplicates —
same defect, same location, same reason → keep one). For each remaining
candidate, run **one verifier** via the `Agent` tool: give it the diff, the
relevant file(s), and the candidate, and have it return exactly one of:

- **CONFIRMED** — can name the inputs/state that trigger it and the wrong
  output or crash. Quote the line.
- **PLAUSIBLE** — mechanism is real, trigger is uncertain (timing, env,
  config). State what would confirm it.
- **REFUTED** — factually wrong (code doesn't say that) or guarded elsewhere.
  Quote the line that proves it.

Keep candidates where the vote is CONFIRMED or PLAUSIBLE. Drop REFUTED.

At **high** effort, verify recall-biased:

> **PLAUSIBLE by default** — do not refute a candidate for being "speculative" or
> "depends on runtime state" when the state is realistic: concurrency races,
> nil/undefined on a rare-but-reachable path (error handler, cold cache, missing
> optional field), falsy-zero treated as missing, off-by-one on a boundary the
> code does not exclude, retry storms / partial failures, regex/allowlist that
> lost an anchor. These are PLAUSIBLE.
>
> **REFUTED** only when constructible from the code: factually wrong (quote the
> actual line); provably impossible (type/constant/invariant — show it); already
> handled in this diff (cite the guard); or pure style with no observable effect.

At **xhigh** and **max**: this is recall mode — a single non-REFUTED vote
carries the finding. Do NOT drop on uncertainty.

## Phase 3 — Sweep for gaps *(xhigh, max)*

Run **one more finder** as a fresh reviewer who has the verified list. Re-read
the diff and enclosing functions looking ONLY for defects not already listed.
Do not re-derive or re-confirm anything already there — the job is gaps. Focus
on what the first pass tends to miss: moved/extracted code that dropped a guard
or anchor; second-tier footguns (dataclass default evaluated once, `hash()`
non-determinism, lock-scope shrink, predicate methods with side effects);
setup/teardown asymmetry in tests; config defaults flipped.

Surface **up to 8 additional candidates**, each naming a defect not already on
the list. If nothing new, return an empty sweep — do not pad.

## Output

Call the `ReportFindings` tool once to report this review's results
with `{level, findings}`. `findings` is at most the level's cap (see the table
above) ranked most-severe first; each entry has `file`, `line`, `summary`,
`short_summary` — the claim compressed to ≤60 characters, no rationale
or consequence clause — `failure_scenario`, and `category` — a short kebab-case slug for the angle
that produced it (`correctness`, `simplification`, `efficiency`,
`reuse`, `altitude`, `conventions`, or a more specific slug like
`test-coverage` when one fits better) — plus `verdict` when a verify pass
produced one. If more than the cap survive, keep the most severe. If
nothing survives verification, call it with an empty array. Do not also print
the findings as text, and do not create or publish an artifact of the review -
the tool call is the report.

If the `ReportFindings` tool is not available, return findings as a JSON array
of at most the level's cap:

```json
[
  {
    "file": "path/to/file.ext",
    "line": 123,
    "summary": "one-sentence statement of the bug",
    "failure_scenario": "concrete inputs/state → wrong output/crash"
  }
]
```

Ranked most-severe first. If more than the cap survive, keep the most severe.
If nothing survives verification, return `[]`.

## Running without the Agent tool

If the `Agent` tool isn't available in this context, the usual multi-agent
fan-out and subagent verify pass can't run. Work through every angle above
yourself, in this same context, in one pass — do not skip angles for lack of
fan-out. Re-check each candidate against the diff before keeping it; drop
anything you can't back up with a concrete failure scenario.

Phase 2 becomes **dedup and self-check, no subagent verify**: dedup
near-duplicates (same defect, same location, same reason → keep one), then
re-check each remaining candidate yourself against the diff before keeping it.
At xhigh and max, still take one more pass yourself as a fresh reviewer holding
the deduplicated list, with the same gap focus as Phase 3.

State clearly in your summary that this was a single-pass review done without
the `Agent` tool, not the full multi-agent fan-out, so whoever reads it isn't
misled about what actually ran.

## Applying fixes (`--fix`)

Only when the `--fix` flag was passed. After producing the findings list, apply
the findings to the working tree instead of stopping at the report: fix each one
directly — correctness bugs and reuse/simplification/efficiency cleanups alike.
Skip any finding whose fix would change intended behavior, require changes well
outside the reviewed diff, or that you judge to be a false positive — note the
skip rather than arguing with it. Then call `ReportFindings` again with the same
findings, each carrying an `outcome`: `fixed`, `no_change_needed` (the finding
was wrong or already handled), or `skipped` (real but not applied). Do not
repeat the findings as text; after the call, give one line per skipped finding
saying why. If `ReportFindings` isn't available, finish with a brief summary of
what was fixed and what was skipped.

Without `--fix`, do not modify any file — the report is the only output.

## Posting to GitHub (`--comment`)

Only when the `--comment` flag was passed. After producing the findings list, if
the review target is a GitHub PR, post each finding as an inline PR comment via
`mcp__github_inline_comment__create_inline_comment` (one call per finding;
include a suggestion block only when it fully fixes the issue). If that tool
is not available in this session, fall back to `gh api` (repos/{owner}/{repo}/pulls/{pr}/comments)
or print the findings instead. If the target is not a PR, print the findings
to the terminal and note that `--comment` was ignored.

## If findings are fixed later

Whenever reported findings get fixed later in this session - the user asks you
to fix them, or later work fixes them incidentally - you MUST call
`ReportFindings` again with the same findings, each carrying an `outcome`:
`fixed`, `no_change_needed` (the finding was wrong or already handled), or
`skipped` (real but not applied). Do not repeat the findings as text.
Make that call immediately after the fixes land, before any prose summary; the
host UI's per-finding status updates only from it, and without it the findings
stay marked unresolved.

## Publishing a shareable review (Artifact)

Only when the review's output contract is *not* the `ReportFindings` tool call
(that contract forbids publishing). Publish the findings as an artifact so they
can be shared and iterated on outside the terminal:

1. Load the `artifact-design` skill (utilitarian treatment —
   this is a document).
2. Write the findings to an HTML file: one section per finding with the file
   path and line, the one-line summary, the concrete failure scenario, and the
   relevant code snippet. If nothing survived verification, the page says so
   in one line.
3. Call the `Artifact` tool with that file path.
4. End the page body with this line verbatim:

   > Paste this URL back into Claude Code to keep iterating on these findings.

Skip this step if the review was invoked only to feed another tool (e.g. a
workflow step whose caller handles its own output).

## After the review

After the findings are reported (and applied, when `--fix` was passed): if
`/verify` has NOT run this session and the diff has a runtime surface (not
test-only or docs-only per the pre-ship exemptions), invoke `/verify` now —
this review checks that the diff reads right; `/verify` checks that it runs
right. State which you did.
