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

#include <cstddef>
#include <cstdint>
#include <vector>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/op/builder/op_builder.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// slice_scatter has no native op: scatter src into the [start:end:step] slice along `dim`.
struct torch_slice_scatter : op_builder<torch_slice_scatter>
{
    int64_t dim   = 0;
    int64_t start = 0;
    int64_t end   = 0;
    int64_t step  = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.dim, "dim"), f(self.start, "start"), f(self.end, "end"), f(self.step, "step"));
    }

    static std::vector<std::string> names() { return {"tm::slice_scatter"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        shape idx_shape{shape::int64_type, args[1]->get_shape().lens()};
        auto axis = tune_axis(idx_shape.ndim(), dim, "slice_scatter");
        std::vector<int64_t> data(idx_shape.elements());
        for(std::size_t i = 0; i < data.size(); ++i)
            data[i] = start + step * idx_shape.multi(i)[axis];
        auto indices = m.add_literal(literal{idx_shape, data.begin(), data.end()});

        auto std_input = m.insert_instruction(ins, make_op("contiguous"), args[0]);
        auto std_src   = m.insert_instruction(ins, make_op("contiguous"), args[1]);
        return {m.insert_instruction(
            ins, make_op("scatter_none", {{"axis", axis}}), {std_input, indices, std_src})};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
