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
#ifndef MIGRAPHX_GUARD_OPERATORS_ZIPMAP_HPP
#define MIGRAPHX_GUARD_OPERATORS_ZIPMAP_HPP

#include <migraphx/shape.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct zipmap
{
    std::vector<int64_t> classlabels_int64s;
    std::vector<std::string> classlabels_strings;

    std::string name() const { return "zipmap"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto input_shape = inputs[0];

        if(input_shape.type() != shape::float_type)
          MIGRAPHX_THROW("ZIPMAP: input must be float type");

        bool has_ints = not classlabels_int64s.empty();
        bool has_strings = not classlabels_strings.empty();

        if (has_ints == has_strings) {
            MIGRAPHX_THROW("ZIPMAP: Must provide keys in either classlabels_strings OR classlabels_int64s.");
        }
        
        std::size_t num_keys = has_ints ? classlabels_int64s.size() : classlabels_strings.size();
        std::size_t vals = input_shape.elements();
        if (num_keys != vals) {
            MIGRAPHX_THROW("ZIPMAP: Number of labels does not match number of input probabilities.");
        }
        shape key_shape;
        if (has_ints) {
            key_shape = shape{shape::int64_type, {num_keys}};
        } else {
            //TODO: support string type when it is implemented.     
            // key_shape = shape{shape::string_type, {0}}; 
        }

        shape output_shape{shape::tuple_type, {key_shape, input_shape}};
        return output_shape;
    }

    argument compute(shape out, std::vector<argument> args) const
    {
        argument result{out};
        auto sub_shapes = out.get_sub_objects();
        auto key_arg = sub_shapes[0];
        auto val_arg = sub_shapes[1];

        if (not classlabels_int64s.empty()) {
            int64_t* key_ptr = key_arg.get_data<int64_t>();
            std::copy(classlabels_int64s.begin(), classlabels_int64s.end(), key_ptr);
        }

        const float* input_ptr = args[0].get_data<float>();
        float* val_ptr = val_arg.get_data<float>();

        std::size_t n = args[0].get_shape().elements();
        std::copy(input_ptr, input_ptr + n, val_ptr);
        
        return result;
    }
};
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif