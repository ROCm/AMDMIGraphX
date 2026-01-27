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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_split_to_sequence : op_parser<parse_split_to_sequence>
{
    std::vector<op_desc> operators() const
    {
        return {{"SplitToSequence"}};
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        // 1. Get the 'axis' attribute (Default 0)
        int64_t axis = 0;
        if(contains(info.attributes, "axis"))
        {
            axis = info.attributes.at("axis").i();
        }

        // 2. Get the 'keepdims' attribute (Default 1)
        int64_t keepdims = 1;
        if(contains(info.attributes, "keepdims"))
        {
            keepdims = info.attributes.at("keepdims").i();
        }

        auto input = args[0];
        auto input_shape = input->get_shape();
        axis          = tune_axis(axis, input_shape.lens().size());
        auto dim_size = input_shape.lens()[axis];
        std::vector<int64_t> split_lengths;

        // 3. Determine Split Lengths
        // Case A: 'split' input is provided
        if(args.size() > 1)
        {
            auto split_arg = args[1];
            // STATIC CHECK: The split configuration MUST be constant
            if(not split_arg->can_eval())
            {
                MIGRAPHX_THROW("PARSE_SPLIT_TO_SEQUENCE: 'split' input must be constant");
            }

            auto split_literal = split_arg->eval();
            // Case A1: 'split' is a list (1D Tensor)
            if(split_literal.get_shape().elements() > 1)
            {
                split_lengths = split_literal.to_vector<int64_t>();
            }
            // Case A2: 'split' is a scalar (equal splits)
            else 
            {
                int64_t split_val = split_literal.at<int64_t>(0);
                
                // Logic from ONNX spec for scalar split
                int64_t num_chunks = dim_size / split_val;
                split_lengths.resize(num_chunks, split_val);
                int64_t remainder = dim_size - (num_chunks * split_val);
                if (remainder > 0) split_lengths.push_back(remainder);
            }
        }
        // Case B: 'split' input is missing (Default: Split into size 1)
        else
        {
            split_lengths.resize(dim_size, 1);
        }

        // 4. Generate the Slice Instructions
        std::vector<instruction_ref> outputs;
        int64_t start_index = 0;

        for(auto len : split_lengths)
        {
            int64_t end_index = start_index + len;
            
            auto slice_op = make_op("slice", {{"axes", {axis}}, 
                                              {"starts", {start_index}}, 
                                              {"ends", {end_index}}});
            
            auto slice_ins = info.add_instruction(slice_op, input);

            // Handle keepdims=0 (Squeeze)
            if(keepdims == 0)
            {
                slice_ins = info.add_instruction(make_op("squeeze", {{"axes", {axis}}}), slice_ins);
            }

            outputs.push_back(slice_ins);
            start_index = end_index;
        }

        // 5. Bundle as a Tuple (The Static Sequence)
        // We use the "identity" operator with multiple inputs to represent a tuple
        return info.add_instruction(make_op("identity"), outputs);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
