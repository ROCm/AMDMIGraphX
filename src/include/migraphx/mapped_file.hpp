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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_MAPPED_FILE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_MAPPED_FILE_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <cstddef>
#include <memory>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct mapped_file_impl;

/**
 * @brief A whole file mapped into memory
 * @details The mapping is private and copy-on-write: writes through data() are never stored in
 * the file. The data stays valid for the lifetime of the object.
 */
struct MIGRAPHX_EXPORT mapped_file
{
    explicit mapped_file(const fs::path& p);
    ~mapped_file();

    mapped_file(const mapped_file&)            = delete;
    mapped_file& operator=(const mapped_file&) = delete;

    /// Start of the mapping; nullptr for an empty file
    char* data() const;
    std::size_t size() const;
    const fs::path& path() const;

    /// Whether any live mapped_file in this process maps the file at p
    static bool is_mapped(const fs::path& p);

    private:
    std::unique_ptr<mapped_file_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
