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
#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_tile : op_parser<parse_tile>
{
    std::vector<op_desc> operators() const { return {{"Tile"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& parser,
                          tf_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        std::vector<int64_t> repeats;

        migraphx::argument arg_s = args[1]->eval();
        if(not arg_s.empty())
            arg_s.visit([&](auto input) { repeats.assign(input.begin(), input.end()); });

        else
        // workaround for dynamic shape, treat tile as a broadcast across the first dimension
        // that is equal to the batch being used for other params
        {
            auto s0 = args[0]->get_shape();
            if(s0.dynamic() or parser.default_dyn_dim_value.max > parser.default_dyn_dim_value.min)
            {
                if(not s0.dynamic())
                    s0 = s0.to_dynamic();
                auto out_dyn_dims = s0.dyn_dims();
                std::vector<size_t> dims_mask(s0.ndim(), 0);
                dims_mask[0] = 1; // TODO find what to set the mask
                auto dims_lit = info.add_literal({{migraphx::shape::int8_type, {s0.ndim()}}, {0*s0.ndim()}});
                return info.add_instruction(make_op("broadcast_with_dims", {{"dims_mask", dims_mask}}), args[0], dims_lit);
            }
            else
            {
                auto out_lens = args[0]->get_shape().lens();
                out_lens[0] = parser.default_dyn_dim_value.max;
                return info.add_instruction(
                make_op("multibroadcast", {{"out_lens", out_lens}}), args[0]);
            }
            
        }


        // Compute output lens by multiplying input dims by repeats
        auto out_lens = args[0]->get_shape().lens();
        if(not repeats.size() == out_lens.size())
        {
            MIGRAPHX_THROW("PARSE_TILE: repeats size mismatch with input shape");
        }
        for(int i = 0; i < out_lens.size(); i++)
        {
            out_lens[i] *= repeats[i];
        }
        
        return info.add_instruction(
            make_op("multibroadcast", {{"out_lens", out_lens}}), args[0]);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
