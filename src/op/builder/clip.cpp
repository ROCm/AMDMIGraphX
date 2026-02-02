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

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct clip : op_builder<clip>
{
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        bool max_used = args.size() == 3 and not args[2]->is_undefined();
        bool min_used = args.size() >= 2 and not args[1]->is_undefined();

        shape input_shape = args.at(0)->get_shape();
        if(min_used and max_used)
        {
            auto bc_args = broadcast_convert_to_shape(m, ins, args, input_shape);
            return {m.insert_instruction(ins, make_op("clip"), bc_args)};
        }
        if(max_used)
        {
            auto bc_args = broadcast_convert_to_shape(m, ins, {args[0], args[2]}, input_shape);
            return {m.insert_instruction(ins, make_op("min"), bc_args)};
        }
        if(min_used)
        {
            auto bc_args = broadcast_convert_to_shape(m, ins, {args[0], args[1]}, input_shape);
            return {m.insert_instruction(ins, make_op("max"), bc_args)};
        }
        return {m.insert_instruction(ins, make_op("identity"), args[0])};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
