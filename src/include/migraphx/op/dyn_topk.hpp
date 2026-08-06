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
#ifndef MIGRAPHX_GUARD_OPERATORS_DYN_TOPK_HPP
#define MIGRAPHX_GUARD_OPERATORS_DYN_TOPK_HPP

#include <algorithm>
#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/op/topk.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * TopK with a data-dependent `k`, matching the ONNX spec where `K` is a runtime input.
 * arg[0]: input data
 * arg[1]: k value, a static 1-D single-element tensor
 *
 * The `k` attribute is the symbol standing for that runtime value, so the output length
 * along `axis` can be described as min(k, axis length) at compile time. The exact length
 * is only known in compute().
 */
struct dyn_topk
{
    sym::expr k{};
    int64_t axis = 0;
    bool largest = true;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.k, "k"), f(self.axis, "axis"), f(self.largest, "largest"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "dyn_topk"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);
        check_shapes{inputs.begin() + 1, inputs.end(), std::string("DYN_TOPK: k input"), false}
            .only_dims(1)
            .elements(1);
        if(k.empty() or k.name() != "variable")
            MIGRAPHX_THROW("DYN_TOPK: k attribute must be a symbolic variable");

        const auto& input_shape = inputs.at(0);
        auto type               = input_shape.type();

        // A range-based axis has no symbolic length to clamp against, so fall back to the
        // widest possible extent.
        // TODO: remove this when range-based dynamic shapes are removed
        if(input_shape.dynamic() and not input_shape.symbolic())
        {
            auto dyn_dims  = input_shape.dyn_dims();
            dyn_dims[axis] = {0, input_shape.max_lens().at(axis)};
            return shape({shape{type, dyn_dims}, shape{shape::int64_type, dyn_dims}});
        }

        auto sym_in    = input_shape.to_symbolic();
        auto dyn_dims  = sym_in.dyn_dims();
        dyn_dims[axis] = shape::dynamic_dimension{sym::min(k, dyn_dims[axis].sym_expr)};
        return shape({shape{type, dyn_dims}, shape{shape::int64_type, dyn_dims}});
    }

    argument compute(const dyn_output&, std::vector<argument> args) const
    {
        auto input_shape  = args.front().get_shape();
        std::size_t k_val = 0;
        args.at(1).visit([&](auto v) { k_val = v.front(); });
        auto actual_k  = std::min<std::size_t>(k_val, input_shape.lens().at(axis));
        auto out_lens  = input_shape.lens();
        out_lens[axis] = actual_k;

        shape out{{shape{input_shape.type(), out_lens}, shape{shape::int64_type, out_lens}}};
        return topk{static_cast<int64_t>(actual_k), axis, largest}.compute(dyn_output{out, out},
                                                                           {args.front()});
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
