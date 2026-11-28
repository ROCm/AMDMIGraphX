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
#include <migraphx/serialize.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/context.hpp>
#include <migraphx/external_data.hpp>
#include <migraphx/mapped_file.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static const external_data* original_external_data(const literal& l)
{
    return l.get_external_data();
}

static const external_data* original_external_data(const argument&) { return nullptr; }

template <class RawData>
static void raw_data_to_value(value& v, const RawData& rd, const serialize_context& sctx)
{
    value result;
    result["shape"] = migraphx::to_value(rd.get_shape());
    if(rd.get_shape().type() == shape::tuple_type)
    {
        result["sub"] = migraphx::to_value(rd.get_sub_objects());
    }
    else if(not rd.empty())
    {
        std::optional<external_data> ext;
        if(sctx.writer != nullptr)
            ext = sctx.writer->write(rd.data(), rd.get_shape().bytes(), original_external_data(rd));
        if(ext.has_value())
            result["external"] = migraphx::to_value(*ext);
        else
            result["data"] = migraphx::value::binary(rd.data(), rd.get_shape().bytes());
    }
    v = result;
}

void migraphx_to_value(value& v, const literal& l) { raw_data_to_value(v, l, {}); }
void migraphx_to_value(value& v, const literal& l, const serialize_context& sctx)
{
    raw_data_to_value(v, l, sctx);
}

void migraphx_from_value(const value& v, literal& l) { migraphx_from_value(v, l, {}); }
void migraphx_from_value(const value& v, literal& l, const serialize_context& sctx)
{
    auto s = migraphx::from_value<shape>(v.at("shape"));
    if(v.contains("data"))
    {
        const auto& data = v.at("data").get_binary();
        if(data.size() != s.bytes())
            MIGRAPHX_THROW("literal: data size " + std::to_string(data.size()) +
                           " does not match shape bytes " + std::to_string(s.bytes()));
        l = literal::from_raw_buffer(s, data.data());
    }
    else if(v.contains("external"))
    {
        if(sctx.reader == nullptr)
            MIGRAPHX_THROW("literal: external data can only be read while loading a program, "
                           "use migraphx::load");
        auto [file, ext] = sctx.reader->read(from_value<external_data>(v.at("external")));
        l                = literal::from_mapped_file(s, std::move(file), std::move(ext));
    }
    else
    {
        l = literal{s.type()};
    }
}

void migraphx_to_value(value& v, const argument& a) { raw_data_to_value(v, a, {}); }
void migraphx_to_value(value& v, const argument& a, const serialize_context& sctx)
{
    raw_data_to_value(v, a, sctx);
}

void migraphx_from_value(const value& v, argument& a) { migraphx_from_value(v, a, {}); }
void migraphx_from_value(const value& v, argument& a, const serialize_context& sctx)
{
    if(v.contains("data") or v.contains("external"))
    {
        literal l = migraphx::from_value<literal>(v, sctx);
        a         = l.get_argument();
    }
    else if(v.contains("sub"))
    {
        // Sub-objects are always written inline
        a = migraphx::from_value<std::vector<argument>>(v.at("sub"));
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
