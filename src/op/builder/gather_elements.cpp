/* The MIT License (MIT)
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

#include <vector>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// gather_elements has no native op: flatten the input and gather element-wise using per-element
// offsets built from the input strides.
struct gather_elements : op_builder<gather_elements>
{
    int64_t axis = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto arg_data = m.insert_instruction(ins, make_op("contiguous"), args[0]);
        auto arg_ind  = m.insert_instruction(ins, make_op("contiguous"), args[1]);

        auto data_s = arg_data->get_shape();
        auto ind_s  = arg_ind->get_shape();
        if(data_s.lens().size() != ind_s.lens().size())
            MIGRAPHX_THROW("gather_elements: input data and index must have the same rank");

        int n_rank       = data_s.lens().size();
        int tuned_axis   = tune_axis(n_rank, axis, "gather_elements");
        auto axis_stride = data_s.strides()[tuned_axis];

        int64_t data_elem_num = data_s.elements();
        arg_data = m.insert_instruction(ins, make_op("reshape", {{"dims", {data_elem_num}}}), arg_data);

        // flat offset of every index position, and its coordinate along the gathered axis
        std::size_t elem_num = ind_s.elements();
        std::vector<int64_t> data_indices(elem_num);
        std::vector<int64_t> axis_indices(elem_num);
        for(std::size_t i = 0; i < elem_num; ++i)
        {
            auto multi      = ind_s.multi(i);
            data_indices[i] = data_s.index(multi);
            axis_indices[i] = multi[tuned_axis];
        }

        auto l_shape_idx = m.add_literal(literal(ind_s, data_indices.begin(), data_indices.end()));
        auto l_dim_idx   = m.add_literal(literal(ind_s, axis_indices.begin(), axis_indices.end()));
        auto l_stride    = m.add_literal(literal{{ind_s.type(), {1}}, {axis_stride}});
        l_stride =
            m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", ind_s.lens()}}), l_stride);

        auto dim_diff = m.insert_instruction(ins, make_op("sub"), arg_ind, l_dim_idx);
        auto delta    = m.insert_instruction(ins, make_op("mul"), dim_diff, l_stride);
        auto ind      = m.insert_instruction(ins, make_op("add"), l_shape_idx, delta);
        return {m.insert_instruction(ins, make_op("gather", {{"axis", 0}}), arg_data, ind)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
