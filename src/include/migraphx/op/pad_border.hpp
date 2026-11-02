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
#ifndef MIGRAPHX_GUARD_OPERATORS_PAD_BORDER_HPP
#define MIGRAPHX_GUARD_OPERATORS_PAD_BORDER_HPP

#include <migraphx/op/pad.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/clamp.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * pad_border(interior, buffer)
 * Writes only the padding region of buffer, which has the shape pad(interior) would
 * produce, and leaves the interior region untouched. A producer can then write its result
 * straight into the interior of buffer instead of copying the whole tensor through a pad.
 */
struct pad_border
{
    op::pad op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "pad_border"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type();
        auto padded = op.compute_shape({inputs.front()});
        if(padded.lens() != inputs.back().lens())
            MIGRAPHX_THROW("PAD_BORDER: buffer dims " + to_string_range(inputs.back().lens()) +
                           " do not match padded dims " + to_string_range(padded.lens()));
        return inputs.back();
    }

    // NOLINTNEXTLINE
    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        const auto& output_shape = dyn_out.computed_shape;
        auto input_lens          = args[0].get_shape().lens();
        auto pad_func            = op.get_pad_func();
        visit_all(args[1], args[0])([&](auto output, auto input) {
            using type = typename decltype(output)::value_type;
            par_for(output_shape.elements(), [&](std::int64_t i) {
                auto out_idx = output_shape.multi(i);
                std::vector<std::int64_t> in_idx(input_lens.size());
                transform(out_idx, op.pads, in_idx.begin(), std::minus<std::int64_t>{});
                // The interior is already in place
                if(migraphx::equal(in_idx, input_lens, std::less<std::size_t>{}))
                    return;
                transform(in_idx, input_lens, in_idx.begin(), pad_func);
                if(migraphx::equal(in_idx, input_lens, std::less<std::size_t>{}))
                    output(out_idx.begin(), out_idx.end()) = input(in_idx.begin(), in_idx.end());
                else
                    output(out_idx.begin(), out_idx.end()) = pad_clamp<type>(op.value);
            });
        });
        return args[1];
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {1}; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
