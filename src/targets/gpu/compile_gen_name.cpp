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

#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

namespace {
std::vector<std::string> get_op_names(const module& m)
{
    std::vector<std::string> result;
    for(auto& ins : m)
    {
        if(starts_with(ins.name(), "@"))
            continue;
        if(contains({"multibroadcast", "contiguous", "identity"}, ins.name()))
            continue;
        if(ins.name() == "pointwise")
        {
            auto names = get_op_names(*ins.module_inputs().front());
            result.insert(result.end(), names.begin(), names.end());
        }
        else
        {
            result.push_back(ins.name());
        }
    }
    return result;
}
} // namespace

std::string generate_name_from_ops(const module& m, const std::string& postname)
{
    auto op_names = get_op_names(m);
    if(not postname.empty())
        op_names.push_back(postname);
    if(op_names.empty())
        return "noop";
    return join_strings(op_names, "_");
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
