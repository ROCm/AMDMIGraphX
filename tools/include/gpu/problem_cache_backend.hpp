/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
//
// te.py DSL for migraphx::gpu::problem_cache_backend.
//
// The generated header lives at
// src/targets/gpu/include/migraphx/gpu/problem_cache_backend.hpp; regenerate it
// with `cd tools && python generate.py` (generate_all routes include/gpu/ inputs
// into the gpu target tree). Do not edit the generated header by hand.
//
// Any type T satisfies the problem_cache_backend concept if it provides the
// member functions listed below. The wrapper holds T by shared_ptr (small
// object optimisation is irrelevant -- backends typically own non-trivial
// resources like file handles or SQLite connections) and forwards each call
// through a virtual dispatch.
//
// Notes:
//   * cache_device_key is defined in <migraphx/gpu/cache_device_key.hpp>;
//     the include below pulls in its full definition.
//   * Backends are NOT expected to be copyable in any meaningful sense
//     beyond what shared ownership provides; they may freely hold streams,
//     file handles, etc.
//   * load() is non-const (mutates internal state); save() is const.
//
#ifndef MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_BACKEND_HPP
#define MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_BACKEND_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/gpu/cache_device_key.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#ifdef DOXYGEN

/// Type-erased interface for problem-cache storage backends.
///
/// A backend persists (cache_device_key, key) -> solution entries to some
/// medium (JSON file, SQLite database, in-memory map for tests, ...). All
/// lookups and inserts are scoped by the device key so a single backend
/// instance can hold entries for multiple GPU configurations.
struct problem_cache_backend
{
    /// Load entries from `path` into the backend. Idempotent; calling
    /// load() twice with the same path leaves the backend in an equivalent
    /// state. Implementations should treat a missing file as "no entries"
    /// rather than an error (the typical first-run case).
    void load(const std::string& path);

    /// Persist all entries to `path`. Const by design: backends that need
    /// to flush write-behind buffers must do so internally without mutating
    /// observable state.
    void save(const std::string& path) const;

    /// Insert (or overwrite) the solution for `key` in the bucket
    /// identified by `dk`. Precondition: `solution` is not null.
    void insert(const cache_device_key& dk, const value& key, const value& solution);

    /// Insert a pending sentinel (null value) for `key`, recording that a
    /// solution for this problem is being tuned. A later lookup that sees the
    /// sentinel treats the problem as already in progress rather than as a
    /// real cached solution.
    void mark(const cache_device_key& dk, const value& key);

    /// Return the stored solution for `key` in the bucket `dk`, or nullopt
    /// if no entry exists. A present optional with a null value indicates
    /// a sentinel (set via mark()).
    optional<value> get(const cache_device_key& dk, const value& key) const;

    /// True iff any entry (real solution or sentinel) exists for `key` in
    /// the bucket `dk`.
    bool has(const cache_device_key& dk, const value& key) const;
};

#else

<%
    interface(
        'problem_cache_backend',
        virtual('load', returns = 'void', path = 'const std::string&'),
        virtual('save', returns = 'void', path = 'const std::string&', const = True),
        virtual('insert',
                returns  = 'void',
                dk       = 'const cache_device_key&',
                key      = 'const value&',
                solution = 'const value&'),
        virtual('mark', returns = 'void', dk = 'const cache_device_key&', key = 'const value&'),
        virtual('get',
                returns = 'optional<value>',
                dk      = 'const cache_device_key&',
                key     = 'const value&',
                const   = True),
        virtual('has',
                returns = 'bool',
                dk      = 'const cache_device_key&',
                key     = 'const value&',
                const   = True))
%>

#endif

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_BACKEND_HPP
