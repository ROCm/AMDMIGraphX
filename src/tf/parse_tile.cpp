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
            // std::cout << "tile shape: " << s0 << std::endl;
            auto base_dyn_dim = shape::dynamic_dimension{1, 1};
            if(s0.dynamic() or parser.default_dyn_dim_value != base_dyn_dim)
            {
                if(not s0.dynamic())
                {
                    s0 = s0.to_dynamic();
                    // std::cout << s0 << std::endl;
                }
                auto out_dyn_dims = s0.dyn_dims();
                out_dyn_dims[0] = parser.default_dyn_dim_value;
                return info.add_instruction(
                    make_op("multibroadcast", {{"out_dyn_dims", to_value(out_dyn_dims)}}), args[0]);
            }
            else
            {
                auto out_lens = args[0]->get_shape().lens();
                out_lens[0] = parser.batch_size;
                return info.add_instruction(
                make_op("multibroadcast", {{"out_lens", out_lens}}), args[0]);
            }
            
        }


        auto l0 = args[0];
        for(int i = 0; i < repeats.size(); i++)
        {
            auto l1 = l0;
            for(int j = 1; j < repeats[i]; j++)
            {
                l0 = info.add_instruction(make_op("concat", {{"axis", i}}), l0, l1);
            }
        }
        return l0;
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
