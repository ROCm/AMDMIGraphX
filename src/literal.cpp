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
#include <migraphx/literal.hpp>
#include <migraphx/external_data.hpp>
#include <migraphx/mapped_file.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

literal
literal::from_mapped_file(const shape& s, std::shared_ptr<mapped_file> file, external_data ext)
{
    if(file == nullptr)
        MIGRAPHX_THROW("literal: no mapped file for external data " + ext.file_path);
    if(s.type() == shape::tuple_type or s.dynamic())
        MIGRAPHX_THROW("literal: external data needs a static non-tuple shape, not " +
                       migraphx::to_string(s));
    if(ext.size != s.bytes())
        MIGRAPHX_THROW("literal: external data size " + std::to_string(ext.size) +
                       " does not match shape bytes " + std::to_string(s.bytes()));
    if(ext.offset > file->size() or ext.size > file->size() - ext.offset)
        MIGRAPHX_THROW("literal: external data of " + std::to_string(ext.size) +
                       " bytes at offset " + std::to_string(ext.offset) +
                       " exceeds size of file: " + file->path().string());
    literal result;
    result.m_shape    = s;
    const char* start = file->data() + ext.offset;
    // The mapping is page aligned, so the offset decides the data's alignment
    if(ext.offset % s.type_size() == 0)
        result.buffer = std::shared_ptr<const char>(file, start);
    else
        result.buffer = make_shared_array<char>(start, start + ext.size);
    result.m_source = std::make_shared<external_data_source>(
        external_data_source{std::move(ext), std::move(file)});
    return result;
}

const external_data* literal::get_external_data() const
{
    return m_source == nullptr ? nullptr : &m_source->ref;
}

std::shared_ptr<mapped_file> literal::get_mapped_file() const
{
    return m_source == nullptr ? nullptr : m_source->file;
}

argument literal::share_argument() const
{
    // argument has no read-only form; its users here only read the data
    auto b = std::const_pointer_cast<char>(buffer); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    return {m_shape, [b]() { return b.get(); }};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
