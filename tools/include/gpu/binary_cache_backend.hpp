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
// te.py DSL for migraphx::gpu::binary_cache_backend.
//
// The generated header lives at
// src/targets/gpu/include/migraphx/gpu/binary_cache_backend.hpp; regenerate it
// with `cd tools && python generate.py` (generate_all routes include/gpu/ inputs
// into the gpu target tree). Do not edit the generated header by hand.
//
// Any type T satisfies the binary_cache_backend concept if it provides the
// member functions listed below. The wrapper holds T by shared_ptr and forwards
// each call through a virtual dispatch, matching problem_cache_backend.
//
// Notes:
//   * binary_cache_entry is defined in <migraphx/gpu/binary_cache_entry.hpp>;
//     the include below pulls in its full definition.
//   * Backends must be copyable: the wrapper shares T and clones it on a
//     non-const call while the handle is shared. sqlite_binary_cache shares its
//     connection across copies.
//
#ifndef MIGRAPHX_GUARD_GPU_BINARY_CACHE_BACKEND_HPP
#define MIGRAPHX_GUARD_GPU_BINARY_CACHE_BACKEND_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <migraphx/config.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/gpu/binary_cache_entry.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#ifdef DOXYGEN

/// Type-erased interface for binary-cache storage backends.
///
/// A backend persists binary_cache_entry values to some medium (a directory of
/// files or a SQLite database), and decides for itself how to serialize them.
/// Entries are addressed by their key, scoped by two strings the caller has
/// already computed:
///
///   * `version` -- binary_cache::version_id(), identifying the toolchain and
///     the embedded kernel sources that produced the entry. Never empty; the
///     caller skips persistence entirely when it is.
///   * `device`  -- the GPU the entry was compiled for.
///
/// A backend must keep entries with different scopes distinct rather than
/// overwriting across them. It may address entries by a hash of the key, for
/// instance to keep file names short, but must then check the full key when
/// loading so that a collision is a miss rather than a wrong kernel.
struct binary_cache_backend
{
    /// Return the entry stored for this key, or nullopt for a miss.
    ///
    /// nullopt also covers every failure: a missing file, an unreadable
    /// database, a damaged entry, a permissions problem. A cache that cannot be
    /// read is not an error, it is a cache miss, and the caller recompiles.
    ///
    /// Must not throw.
    optional<binary_cache_entry>
    load(const std::string& version, const std::string& device, const std::string& key);

    /// Persist every entry in `entries` under its key. The entries arrive
    /// together so a backend can commit them at once, such as in one database
    /// transaction, rather than one at a time.
    ///
    /// Overwriting an existing entry is expected and safe: the content is
    /// decided entirely by the key, so a writer that loses a race replaces the
    /// entry with an equivalent one.
    ///
    /// May throw: the caller reports a failed store as a warning. It costs a
    /// recompile next run, nothing more, and the caller still keeps the results
    /// in memory. A backend that throws must not leave anything locked.
    void store(const std::string& version,
               const std::string& device,
               const std::vector<binary_cache_entry>& entries);
};

#else

<%
    interface('binary_cache_backend',
              virtual('load',
                      returns = 'optional<binary_cache_entry>',
                      version = 'const std::string&',
                      device  = 'const std::string&',
                      key     = 'const std::string&'),
              virtual('store',
                      returns = 'void',
                      version = 'const std::string&',
                      device  = 'const std::string&',
                      entries = 'const std::vector<binary_cache_entry>&'))
%>

#endif

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_BINARY_CACHE_BACKEND_HPP
