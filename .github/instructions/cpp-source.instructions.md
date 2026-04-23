---
applyTo: "**/*.{cpp,hpp,h,cc,cxx}"
---

# C++ Review Instructions for MIGraphX

Review C++ changes against the MIGraphX coding style and flag AI-generated slop.

## Coding Style

### Naming & Syntax
- `snake_case` for files, functions, variables, types
- `CamelCase` for template parameters
- `UPPER_CASE` with `MIGRAPHX_` prefix for macros
- `using alias = T;` not `typedef`
- `nullptr` not `NULL` / `0`
- `and` / `or` instead of `&&` / `||`
- 4-space indent, 100-column limit, braces on new line (per `.clang-format`)

### Types & Casts
- Declare the correct type at the source. Flag any `static_cast` / `reinterpret_cast` / C-style cast that exists only to paper over a type mismatch.
- Prefer `std::bitset` and named utility helpers over raw bit twiddling.
- Use `std::tie` or `std::lexicographical_compare` for lexicographic comparisons.
- Use `migraphx::shape` methods for offsets, strides, and indexing — never hand-rolled mod/div arithmetic.

### Algorithms Over Raw Loops
- Replace raw `for` loops with `<algorithm>` or `migraphx/algorithm.hpp` (`transform_if`, `transform_accumulate`, `group_by`, `adjacent_for_each`, etc.).
- `std::for_each` is NOT an acceptable substitute — it's just a loop wrapper.
- If no suitable algorithm exists, suggest adding one to `migraphx/algorithm.hpp` rather than accepting a raw loop.

### Resources & Memory
- `std::make_unique` / `std::make_shared`; no raw `new` / `delete`.
- Use `MIGRAPHX_MANAGE_PTR` for C-style acquire/release APIs.

### Type Erasure
- `pass`, `operation`, `target`, `instruction` use type erasure — do not add inheritance. Implement the required methods directly on a value type.
- Interface changes require running `make generate`; flag hand-edits to generated headers.

### Passes
- Must be idempotent and deterministic.
- `compute_shape` must handle dynamic (unknown) dimensions.
- Prefer matcher-based rewrites (`src/matcher.hpp`) over open-coded traversal.

## AI Slop — Flag and Suggest Removal

Scan for these patterns; they are common in AI-generated C++ and rarely belong in a PR:

1. **Redundant comments** describing *what* the code does (`// increment i`, `// return the result`). Keep comments only for non-obvious *why* — hidden constraints, workarounds, invariants. Delete the rest.
2. **Multi-line docstrings / banner comments** on trivial functions. One short line max.
3. **References to the task or PR in comments** (`// added for issue #123`, `// fix from review`). Belongs in the commit message, not the code.
4. **Defensive checks at internal boundaries** — null checks, `if (vec.empty()) return;`, try/catch around infallible code. Only validate at true system boundaries.
5. **Speculative error handling** for conditions that cannot occur given caller guarantees.
6. **Backwards-compat shims** that aren't needed: renamed `_unused` parameters, re-exports of removed types, `// removed` tombstone comments.
7. **Unrequested refactors** bundled into a bug fix. Flag them for separation.
8. **Unused casts, unused `std::move`, unnecessary `auto&&`**, copies where a reference suffices, `const_cast` used to bypass API design.
9. **Magic numbers and duplicated literals** where a named constant or existing enum already exists.
10. **Tests that assert what the code just did** rather than observable behavior; overuse of mocks where a real object is cheap.
11. **Over-broad `#include`s** — including a heavy header when a forward declaration or a narrower header would do.

## Overengineering — Flag Aggressively

AI-generated code routinely builds machinery the task did not ask for. Push back on all of these:

- **Premature abstraction** — a helper used once, a template parameter with one instantiation, a base class with one derived class, a policy/strategy class for two branches. Three similar lines are better than a wrong abstraction.
- **Classes masquerading as functions** — a struct with one public method and no state should be a free function.
- **Unnecessary templates** — templating a function that only ever takes `instruction_ref` or `std::string`. Use the concrete type.
- **Needless virtual / inheritance hierarchies** in a codebase that uses type erasure (`pass`, `operation`, `target`). Flag any new `virtual` or base class added to these areas.
- **Config structs / builder patterns** around what should be 1–2 function arguments.
- **Factory functions** that only call a constructor.
- **Custom exception types, custom iterators, custom smart pointers** when the standard or existing MIGraphX equivalent works.
- **Multiple layers of wrappers** (e.g., `wrap_impl` calling `do_wrap` calling `wrap_inner`) with no distinct responsibility per layer. Collapse them.
- **"Future-proofing"** — enums with a single value, `std::variant` with one alternative, hooks/callbacks with no second caller, `std::optional` return where the value is always present.
- **Parallel data structures** — a new `std::map<instruction_ref, X>` sidecar where an existing field on `instruction` or `module` carries the same information.
- **New files for 10-line helpers** that belong next to their one caller.

When you see it, propose the concrete collapse: "inline this into the caller," "delete the template and pass `shape` directly," "use the existing field on `instruction` instead of a new map."

## Reuse Existing MIGraphX Utilities

AI-generated code loves to re-implement things MIGraphX already provides. Before accepting any hand-rolled helper, check whether it duplicates one of these. If it does, the PR should use the existing utility:

**Algorithms & ranges**
- `<migraphx/algorithm.hpp>` — `transform_if`, `transform_accumulate`, `transform_partial_sum`, `group_by`, `group_unique`, `group_find`, `adjacent_remove_if`, `adjacent_for_each`, two-range `for_each`, `min_element_if`, `levenshtein_distance`.
- `<migraphx/ranges.hpp>` — `contains`, `all_of`, `any_of`, `none_of`, `generic_find` on whole containers; stop writing `std::find(c.begin(), c.end(), x) != c.end()`.
- `<migraphx/iterator_for.hpp>` — iterate instructions in a module without raw iterator arithmetic.
- `<migraphx/erase.hpp>` — uniform container erase; no hand-rolled erase-remove idiom.
- `<migraphx/output_iterator.hpp>` — function-object output iterators instead of bespoke wrappers.

**Shape / tensor math**
- `migraphx::shape` — `.index()`, `.multi()`, `.element_space()`, `.packed()`, `.transposed()`, `.broadcasted()`, `.standard()`. Never hand-code stride/offset arithmetic with mod/div.
- `<migraphx/common.hpp>`, `<migraphx/common_dims.hpp>` — broadcasting, shape-unification helpers.
- `<migraphx/check_shapes.hpp>` — shape validation in `compute_shape`; don't inline the same `if (inputs.size() != N)` checks.
- `<migraphx/dfor.hpp>` — N-dimensional iteration.
- `<migraphx/clamp.hpp>`, `<migraphx/float_equal.hpp>` — numeric comparisons; do not reinvent `abs(a-b) < eps`.

**IR construction & inspection**
- `<migraphx/make_op.hpp>` — `make_op("name", {{"attr", val}})`; don't construct op structs by name-string concatenation.
- `<migraphx/matcher.hpp>` — pattern-based rewrites (`match::name`, `match::args`, `match::either_arg`, `.bind`). Flag any hand-written DFS that re-implements matching.
- `module::replace_instruction`, `module::insert_instruction`, `module::remove_instruction` — never splice the instruction list directly.

**Strings, hashing, I/O**
- `<migraphx/stringutils.hpp>` — `replace_string`, `starts_with`, `ends_with`, `contains`, `split_string`, `join_strings`, `trim`, `to_upper/lower`.
- `<migraphx/hash.hpp>` — combine hashes via the provided helpers, not ad-hoc `x ^ (y << 1)`.
- `<migraphx/file_buffer.hpp>`, `<migraphx/filesystem.hpp>`, `<migraphx/fileutils.hpp>` — file I/O and path handling.
- `<migraphx/errors.hpp>` — `MIGRAPHX_THROW`; do not `throw std::runtime_error` directly.
- `<migraphx/assert.hpp>` — `MIGRAPHX_ASSERT` over raw `assert`.
- `<migraphx/env.hpp>` — environment variables via the provided accessors.

**Resources & memory**
- `MIGRAPHX_MANAGE_PTR` for C-style acquire/release APIs.
- `<migraphx/any_ptr.hpp>`, `<migraphx/bit_cast.hpp>`, `<migraphx/functional.hpp>` — prefer these over hand-rolled equivalents.

When reviewing, if a PR adds a free function, lambda, or helper struct that looks like one of the above, link to the existing header and ask for it to be replaced. A 30-line "utility" added in a feature PR is almost always a reinvented wheel.

## Review Output

For each issue, suggest the concrete simplification (delete / replace with algorithm X / use existing utility Y) rather than a generic "consider refactoring." If you can point at a specific MIGraphX header that already solves the problem, name it.
