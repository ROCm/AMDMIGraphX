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
#include <migraphx/gpu/check_recompiled_memory.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static std::string describe(instruction_ref ins)
{
    if(ins->name() == "gpu::code_object")
        return ins->name() + "[" + any_cast<code_object_op>(ins->get_operator()).symbol_name + "]";
    return ins->name();
}

void check_recompiled_memory::apply(const module& m) const
{
    std::vector<std::string> users;
    for(const auto& ins : m)
    {
        if(ins.name() != "hip::allocate")
            continue;
        std::transform(ins.outputs().begin(),
                       ins.outputs().end(),
                       std::back_inserter(users),
                       [](instruction_ref out) { return describe(out); });
    }
    if(not users.empty())
        MIGRAPHX_THROW("recompile: the recompiled kernels need memory that the compiled program "
                       "does not have (used by " +
                       join_strings(users, ", ") + "); compile the model again");
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
