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
#ifndef MIGRAPHX_GUARD_RTGLIB_LOAD_SAVE_HPP
#define MIGRAPHX_GUARD_RTGLIB_LOAD_SAVE_HPP

#include <migraphx/program.hpp>
#include <migraphx/external_data.hpp>
#include <optional>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct file_options
{
    std::string format = "msgpack";
    /// Load the kernels of a compiled program. When false, the program can be opened on a
    /// machine without the device it was compiled for, for example to recompile it.
    bool finalize = true;
    /// Save literals of at least this many bytes to an external weights file; nullopt keeps
    /// them inline
    std::optional<std::size_t> external_weights_threshold = std::nullopt;
    /// Save a reference to the file a literal was read from (an ONNX or weights file) instead of
    /// its data
    bool reuse_external_data = false;
    /// Base name of the weights file: each save writes a new generation beside it, named
    /// <stem>.<id><extension>; empty uses the program file name + external_data_extension
    std::string external_weights_file;
    /// Byte alignment of each literal in the weights file; must be a nonzero power of two
    std::size_t external_weights_alignment = external_data_alignment;
    /// Directory relative external references resolve against, and which every reference must
    /// lie within; empty uses the program file's directory
    std::string external_data_path;
    /// Further directories external references may point into when loading
    std::vector<std::string> external_data_allowed_paths;
    /// Check the md5 checksum of each external literal when loading, rejecting literals that
    /// have none; this detects corruption, not tampering
    bool verify_external_weights = false;
};

MIGRAPHX_EXPORT program load(const std::string& filename,
                             const file_options& options = file_options{});
MIGRAPHX_EXPORT program load_buffer(const std::vector<char>& buffer,
                                    const file_options& options = file_options{});
MIGRAPHX_EXPORT program load_buffer(const char* buffer,
                                    std::size_t size,
                                    const file_options& options = file_options{});

/// With external weights, the new weights generation is published before the program file, and
/// the generations the replaced program file referenced are then removed unless still in use
MIGRAPHX_EXPORT void
save(const program& p, const std::string& filename, const file_options& options = file_options{});
/// With external weights, writes a new weights generation that is never removed by a later save
MIGRAPHX_EXPORT std::vector<char> save_buffer(const program& p,
                                              const file_options& options = file_options{});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
