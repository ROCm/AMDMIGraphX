/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <limits>
#include <migraphx/serialize.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class RawData>
void raw_data_to_value(value& v, const RawData& rd)
{
    value result;
    result["shape"] = migraphx::to_value(rd.get_shape());
    if(rd.get_shape().type() == shape::tuple_type)
        result["sub"] = migraphx::to_value(rd.get_sub_objects());
    else if(not rd.empty())
    {
        size_t binary_size      = rd.get_shape().bytes();
        size_t partition_length = std::numeric_limits<uint32_t>::max();
        if(binary_size > partition_length)
        {
            size_t array_size = 1 + ((binary_size - 1) / partition_length);
            std::vector<migraphx::value> v_array(array_size);
            for(size_t i = 0; i < array_size; ++i)
            {
                size_t chunk_size =
                    (i == (array_size - 1)) ? (binary_size % partition_length) : partition_length;
                v_array[i] =
                    migraphx::value::binary{(rd.data() + (i * partition_length)), chunk_size};
            }
            result["data"] = migraphx::value(v_array);
        }
        else
        {
            result["data"] = migraphx::value::binary(rd.data(), rd.get_shape().bytes());
        }
    }
    v = result;
}

void migraphx_to_value(value& v, const literal& l) { raw_data_to_value(v, l); }
void migraphx_from_value(const value& v, literal& l)
{
    auto s                  = migraphx::from_value<shape>(v.at("shape"));
    size_t binary_size      = s.bytes();
    size_t partition_length = std::numeric_limits<uint32_t>::max();
    if(binary_size <= partition_length)
    {
        l = literal(s, v.at("data").get_binary().data());
    }
    else
    {
        auto v_data = v.at("data");
        if(not v_data.is_array())
        {
            MIGRAPHX_THROW("Literal is larger than 4GB but it is not stored as binary array");
        }
        size_t array_size = 1 + ((binary_size - 1) / partition_length);
        std::vector<uint8_t> binary_array(binary_size);
        size_t read_size = 0;
        for(size_t i = 0; i < array_size; ++i)
        {
            binary_array.insert(binary_array.end(),
                                v_data.at(i).get_binary().data(),
                                v_data.at(i).get_binary().data() +
                                    v_data.at(i).get_binary().size());
            read_size += v_data.at(i).get_binary().size();
        }
        if(read_size != binary_size or array_size != v_data.size())
        {
            MIGRAPHX_THROW("Literal deserialization failed. File is corrupted");
        };
        l = literal(s, binary_array.data());
    }
}

void migraphx_to_value(value& v, const argument& a) { raw_data_to_value(v, a); }
void migraphx_from_value(const value& v, argument& a)
{
    if(v.contains("data"))
    {
        literal l = migraphx::from_value<literal>(v);
        a         = l.get_argument();
    }
    else if(v.contains("sub"))
    {
        a = migraphx::from_value<std::vector<argument>>(v.at("sub"));
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
