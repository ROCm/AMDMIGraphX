/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_gather_elements : op_parser<parse_gather_elements>
{
    std::vector<op_desc> operators() const { return {{"GatherElements"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        int axis = 0;
        if(contains(info.attributes, "axis"))
        {
            axis = parser.parse_value(info.attributes.at("axis")).at<int>();
        }

        // standardize input data and index
        auto arg_data = info.make_contiguous(args[0]);
        auto arg_ind  = info.make_contiguous(args[1]);

        auto data_s = arg_data->get_shape();
        auto ind_s  = arg_ind->get_shape();

        if(data_s.ndim() != ind_s.ndim())
        {
            MIGRAPHX_THROW("PARSE_GATHER_ELEMENTS: input data and index must have the same rank!");
        }

        int n_rank     = data_s.ndim();
        int tuned_axis = tune_axis(n_rank, axis, opd.onnx_name);

        if(data_s.dynamic())
        {
            MIGRAPHX_THROW("PARSE_GATHER_ELEMENTS: dynamic input data is not supported");
        }

        auto axis_stride      = data_s.strides()[tuned_axis];
        int64_t data_elem_num = data_s.elements();
        // reshape the input data as one dimension and used as input data
        // to the gather operator
        arg_data = info.add_instruction(make_op("reshape", {{"dims", {data_elem_num}}}), arg_data);

        if(ind_s.dynamic())
        {
            // The index into the flattened data is base + index * axis_stride, where base is
            // the offset of the other coordinates. It needs their lens, so only the axis may vary.
            auto dds = ind_s.dyn_dims();
            std::vector<std::size_t> base_lens(dds.size(), 1);
            for(int i = 0; i < n_rank; i++)
            {
                if(i == tuned_axis)
                    continue;
                if(not dds[i].is_fixed())
                    MIGRAPHX_THROW("PARSE_GATHER_ELEMENTS: index can only be dynamic on the axis");
                base_lens[i] = dds[i].get_interval().min;
            }
            shape base_s{ind_s.type(), base_lens};
            std::vector<int64_t> base(base_s.elements());
            for(std::size_t i = 0; i < base.size(); i++)
                base[i] = data_s.index(base_s.multi(i));
            auto l_base   = info.add_literal(literal(base_s, base.begin(), base.end()));
            auto l_stride = info.add_literal(literal{{ind_s.type(), {1}}, {axis_stride}});
            auto delta    = info.add_common_op("mul", arg_ind, l_stride);
            auto ind      = info.add_common_op("add", l_base, delta);
            return info.add_instruction(make_op("gather", {{"axis", 0}}), arg_data, ind);
        }

        std::size_t elem_num = ind_s.elements();
        std::vector<int> ind_index(elem_num);
        std::iota(ind_index.begin(), ind_index.end(), 0);

        // convert index in input indices to that in input data
        std::vector<int> data_indices(elem_num);
        std::transform(ind_index.begin(), ind_index.end(), data_indices.begin(), [&](auto i) {
            return data_s.index(ind_s.multi(i));
        });

        std::vector<int> vec_axis_ind(elem_num);
        std::transform(ind_index.begin(), ind_index.end(), vec_axis_ind.begin(), [&](auto i) {
            return ind_s.multi(i)[tuned_axis];
        });

        auto l_shape_idx =
            info.add_literal(literal(ind_s, data_indices.begin(), data_indices.end()));
        auto l_dim_idx = info.add_literal(literal(ind_s, vec_axis_ind.begin(), vec_axis_ind.end()));
        auto l_stride  = info.add_literal(literal{{ind_s.type(), {1}}, {axis_stride}});
        l_stride =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", ind_s.lens()}}), l_stride);
        auto dim_diff = info.add_instruction(make_op("sub"), arg_ind, l_dim_idx);
        auto delta    = info.add_instruction(make_op("mul"), dim_diff, l_stride);
        auto ind      = info.add_instruction(make_op("add"), l_shape_idx, delta);

        auto op = make_op("gather", {{"axis", 0}});
        return info.add_instruction(op, arg_data, ind);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
