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

#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/broadcast_dimensions.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

namespace detail
{
    std::vector<instruction_ref>
    insert(module& m, const std::string& name, const std::vector<instruction_ref>& args)
    {
        auto a0      = args[0];
        auto a1      = args[1];
        auto ba0     = args.size() > 2 ? args[2] : a0;
        auto ba1     = args.size() > 3 ? args[3] : a1;

        op::builder::broadcast_dimensions(m, a0, a1, ba0, ba1);
        return {m.add_instruction(make_op(name), ba0, ba1)};
    }
}

struct dot : op_builder<dot>
{
    std::vector<instruction_ref>
    insert(module& m, instruction_ref /*ins*/, const std::vector<instruction_ref>& args) const
    {
        return detail::insert(m, name(), args);
    }
};

struct quant_dot : op_builder<quant_dot>
{
    std::vector<instruction_ref>
    insert(module& m, instruction_ref /*ins*/, const std::vector<instruction_ref>& args) const
    {
        return detail::insert(m, name(), args);
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
