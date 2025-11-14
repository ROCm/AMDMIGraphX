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

#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct tile : op_builder<tile>
{
    std::vector<std::int64_t> repeats;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.repeats, "repeats"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref /*ins*/, const std::vector<instruction_ref>& args) const
    {
        const auto& input_shape = args[0]->get_shape();
        const auto& input_lens  = input_shape.lens();

        if(not(repeats.size() == input_lens.size()))
        {
            MIGRAPHX_THROW("tile op-builder: repeats size mismatch with input shape");
        }

        /*
        input_lens:           {l0, l1, l2, ..., lN-1} - size: N
        repeats:              {r0, r1, r2, ..., rN-1} - size: N
        after unsqueeze:      {1, l0, 1, l1, 1, l2, ..., 1, lN-1}; - size: 2*N; putting 1 before each dimension
        after multibroadcast: {r0, l0, r1, l1, r2, l2, ..., rN-1, lN-1}; - size: 2*N; putting r_i before each dimension
        after reshape:        {r0*l0, r1*l1, r2*l2, ..., rN-1*lN-1}; - size: N again; multiplying each pair of dimensions
        */

        std::vector<int64_t> unsq_axes(input_lens.size());
        std::iota(unsq_axes.begin(), unsq_axes.end(), 0);
        std::transform(
            unsq_axes.begin(), unsq_axes.end(), unsq_axes.begin(), [](auto x) { return 2 * x; });

        auto unsq =
            m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsq_axes}}), args[0]);

        std::vector<std::size_t> bcast_shape_lens = unsq->get_shape().lens();
        std::for_each(unsq_axes.begin(), unsq_axes.end(), [&](int64_t axis_idx) {
            bcast_shape_lens[axis_idx] = repeats[axis_idx / 2];
        });
        migraphx::shape bcast_shape{input_shape.type(), bcast_shape_lens};

        auto mbcast = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", bcast_shape.lens()}}), unsq);

        std::vector<std::size_t> reshape_dims(bcast_shape_lens.size() / 2);
        for(size_t i = 0; i < reshape_dims.size(); i++)
        {
            reshape_dims[i] = bcast_shape_lens[i * 2] * bcast_shape_lens[i * 2 + 1];
        }

        return {m.add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims}}), mbcast)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
