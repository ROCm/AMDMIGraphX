/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_QUANT_DOT_HPP
#define MIGRAPHX_GUARD_OPERATORS_QUANT_DOT_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gemm.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/value.hpp>
#include <migraphx/fp8_types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * 2 input version:
 * Standard quantized GEMM operation
 * inputs = {A_mat, B_mat}
 *
 * 4 input version:
 * Quantized GEMM with two sets of scales for A and B matricies.
 * inputs = {A_mat, B_mat, scale_A, scale_B}
 */
struct quant_dot
{
    value attributes() const { return {{"general_data_type", "dot"}}; }

    std::string name() const { return "quant_dot"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2, 4);
        check_shapes{{inputs.at(0), inputs.at(1)}, *this, true}.same_type().same_ndims();
        if(inputs.size() == 4)
        {
            check_shapes{{inputs.at(2), inputs.at(3)}, *this, true}.same_type();
        }
        const shape& a                                    = inputs.at(0);
        const shape& b                                    = inputs.at(1);
        auto t                                            = a.type();
        std::set<migraphx::shape::type_t> supported_types = fp8_types{}.get();
        supported_types.insert(shape::int8_type);
        supported_types.insert(shape::uint8_type);
        // for how mxfp4 is handled with pack/unpack
        supported_types.insert(shape::float_type);
        if(not contains(supported_types, t))
        {
            MIGRAPHX_THROW("QUANT_DOT: only supports int8_t, uint8_t, float, and fp8");
        }
        if(not std::all_of(inputs.begin(), inputs.end(), [](auto s) { return s.ndim() >= 2; }))
        {
            MIGRAPHX_THROW("QUANT_DOT: dot only accepts >= 2D operands");
        }

        auto out_type = (inputs.size() == 4 or contains(fp8_types{}.get(), t)) ? shape::float_type
                                                                               : shape::int32_type;

        if(a.dynamic() or b.dynamic())
        {
            auto s0 = a.to_dynamic();
            auto s1 = b.to_dynamic();
            std::vector<shape::dynamic_dimension> out_dyn_dims;

            bool same_outers = std::equal(s0.dyn_dims().begin(),
                                          s0.dyn_dims().end() - 2,
                                          s1.dyn_dims().begin(),
                                          s1.dyn_dims().end() - 2,
                                          [&](const auto& x, const auto& y) {
                                              auto intersect = x.intersection(y);
                                              if(intersect.has_value())
                                              {
                                                  out_dyn_dims.push_back(intersect.value());
                                                  return true;
                                              }
                                              return false;
                                          });
            if(not same_outers)
            {
                MIGRAPHX_THROW("QUANT_DOT: dynamic outer dimensions of A and B mismatch: {" +
                               to_string_range(s0.dyn_dims()) + "} x {" +
                               to_string_range(s1.dyn_dims()) + "}");
            }
            std::size_t dim_i = s0.ndim() - 2;
            std::size_t dim_j = s0.ndim() - 1;
            if(not s0.dyn_dims()[dim_j].intersection(s1.dyn_dims()[dim_i]).has_value())
            {
                MIGRAPHX_THROW("QUANT_DOT: dynamic inner dimensions are not compatible: {" +
                               to_string_range(s0.dyn_dims()) + "} x {" +
                               to_string_range(s1.dyn_dims()) + "}");
            }
            out_dyn_dims.push_back(s0.dyn_dims()[dim_i]);
            out_dyn_dims.push_back(s1.dyn_dims()[dim_j]);
            return {out_type, out_dyn_dims};
        }

        if(not std::equal(
               a.lens().rbegin() + 2, a.lens().rend(), b.lens().rbegin() + 2, b.lens().rend()))
        {
            MIGRAPHX_THROW("QUANT_DOT: batch size of A and B mismatch: {" +
                           to_string_range(a.lens()) + "} x {" + to_string_range(b.lens()) + "}");
        }

        std::size_t dim_0 = a.lens().size() - 2;
        std::size_t dim_1 = a.lens().size() - 1;
        if(a.lens()[dim_1] != b.lens()[dim_0])
        {
            MIGRAPHX_THROW("QUANT_DOT: inner dimensions do not match: {" +
                           to_string_range(a.lens()) + "} x {" + to_string_range(b.lens()) + "}");
        }

        auto out_lens   = a.lens();
        out_lens[dim_1] = b.lens()[dim_1];
        return {out_type, out_lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
