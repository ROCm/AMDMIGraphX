---
applyTo: "**/*.{cpp,hpp,h,cc,cxx}"
---

# C++ Review Instructions for MIGraphX

Review C++ changes against the MIGraphX coding style and flag AI-generated slop.

## Coding Style

### Naming Conventions

- **Files and symbols**: `snake_case` (e.g., `compute_shape`, `eliminate_concat.cpp`)
- **Template parameters**: `CamelCase`
- **Macros**: `UPPER_CASE` with `MIGRAPHX_` prefix
- Use `using type_alias = int` not `typedef int type_alias`
- Use `nullptr` not `NULL` or `0`
- Use `or` and `and` instead `&&` and `||`

### Formatting (.clang-format)

- 4-space indent, no tabs
- 100 column limit
- Braces on new line for classes/functions/control structures
- Align consecutive assignments

### Style Guidelines

- Avoid raw loops - Prefer algorithms from `<migraphx/algorithm.hpp>` and STL `<algorithm>`:
    - `std::for_each` is NOT an acceptable substitute — it's just a loop wrapper.
    - MIGraphX-specific algorithms (`#include <migraphx/algorithm.hpp>`):
      - `transform_if(start, last, out, pred, f)` - Transform with filtering
      - `transform_accumulate(first, last, init, binop, unaryop)` - Accumulate with projection
      - `transform_partial_sum(first, last, out, binop, unaryop)` - Partial sum with projection
      - `group_by(start, last, out, pred)` - Group elements by predicate
      - `group_unique(start, last, out, pred)` - Group consecutive unique elements
      - `group_find(start, last, pred, out)` - Find and group matching ranges
      - `adjacent_remove_if(first, last, pred)` - Remove based on adjacent pairs
      - `adjacent_for_each(first, last, f)` - Iterate over adjacent pairs
      - `for_each(first1, last1, first2, f)` - Two-range iteration (like std::transform)
    - If no suitable algorithm exists: Add a new algorithm to `migraphx/algorithm.hpp` rather than using raw loops

- Memory management - Use `std::make_unique/shared`, avoid raw `new`/`delete`
- Non-memory resources - Use `MIGRAPHX_MANAGE_PTR` for C-style acquire/release APIs:
- Type erasure over inheritance - MIGraphX uses type erasure extensively for `pass`, `operation`, `target`
  - No need to inherit from base class - just implement required interface methods
  - Generate boilerplate with `make generate` when interfaces change
  - Enables value semantics with polymorphism, no virtual inheritance overhead

- Pass design principles:
  - Keep passes idempotent (running twice = running once)
  - Keep passes deterministic (same input → same output)
  - Always handle dynamic shapes in `compute_shape` (unknown dimensions)
  - Prefer matcher-based rewrites (`src/matcher.hpp`) over open-coded traversal.

- Generic programming
    - Write reusable, type-independent code using templates and STL algorithms

- Avoid Casts
    - Dont use a cast unless abosuluty necessary. Declare Correct Types. Casts indicate a type mismatch that should be resolved at the source.

- Encapsulate Bit Manipulation - Put bit-twiddling and low-level operations behind well-named utility functions:
    - Prefer `std::bitset` for bit manipulation

- Use std::tie for Lexicographical Comparisons
    - Use `std::tie` or `std::lexicographical_compare` instead of manually writing lexicographical comparisons.

- Use shape class to compute offsets and indexing
    - The `migraphx::shape` class provides methods for computing offsets, strides, and indexing. Use these instead of manual calculations with mod and division.

## AI slop

- **Redundant comments** describing *what* the code does (`// increment i`, `// return the result`). Keep comments only for non-obvious *why* — hidden constraints, workarounds, invariants. Delete the rest.
- **Multi-line docstrings / banner comments** on trivial functions. One short line max.
- **Defensive checks at internal boundaries** — null checks, `if (vec.empty()) return;`, try/catch around infallible code. Only validate at true system boundaries.
- **Speculative error handling** for conditions that cannot occur given caller guarantees.
- **Backwards-compat shims** that aren't needed: renamed `_unused` parameters, re-exports of removed types, `// removed` tombstone comments.
- **Unrequested refactors** bundled into a bug fix. Flag them for separation.
- **Unused casts, unused `std::move`, unnecessary `auto&&`**, copies where a reference suffices, `const_cast` used to bypass API design.
- **Magic numbers and duplicated literals** where a named constant or existing enum already exists.
- **Over-broad `#include`s** — including a heavy header when a forward declaration or a narrower header would do.
- **Premature abstraction** — a helper used once
- **Classes masquerading as functions** — a struct with one public method and no state should be a free function.
- **Config structs / builder patterns** around what should be 1–2 function arguments.
- **Factory functions** that only call a constructor.
- **Multiple layers of wrappers** (e.g., `wrap_impl` calling `do_wrap` calling `wrap_inner`) with no distinct responsibility per layer. Collapse them.
- **"Future-proofing"** — enums with a single value, `std::variant` with one alternative, hooks/callbacks with no second caller, `std::optional` return where the value is always present.
- **Unnecessary intermediate variables** - When the original variable can just be used directly

Look for ways to simplify the code by resusing existing migraphx utilites.
