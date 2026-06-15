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

// The MLIR backend plugin links migraphx core + rocMLIR but NOT migraphx_gpu
// (that would collide with the public mlir.hpp symbols the plugin defines).
// mlir.cpp pulls a single non-inline helper from compile_gen.cpp, which is
// trivial, so it is reimplemented here to keep the plugin self-contained.

#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/errors.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

namespace {
std::vector<std::string> plugin_get_op_names(const module& m)
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
            auto names = plugin_get_op_names(*ins.module_inputs().front());
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
    auto op_names = plugin_get_op_names(m);
    if(not postname.empty())
        op_names.push_back(postname);
    if(op_names.empty())
        return "noop";
    return join_strings(op_names, "_");
}

} // namespace gen

// code_object_op's behaviour (compute_shape/compute/finalize) lives in
// migraphx_gpu (code_object_op.cpp, which also registers the op). The plugin
// only ever constructs code_object_op as plain data and hands it back to the
// host backend loader, which performs the type-erasure/insertion. These definitions exist
// solely to satisfy the operation<> instantiation pulled in by the plugin's
// (unused) copy of insert_mlir; they must never run inside the plugin.
shape code_object_op::compute_shape(std::vector<shape>) const
{
    MIGRAPHX_THROW("code_object_op::compute_shape must not be called inside an MLIR backend plugin");
}

argument code_object_op::compute(context&, const shape&, const std::vector<argument>&) const
{
    MIGRAPHX_THROW("code_object_op::compute must not be called inside an MLIR backend plugin");
}

void code_object_op::finalize(context&, const shape&, const std::vector<shape>&)
{
    MIGRAPHX_THROW("code_object_op::finalize must not be called inside an MLIR backend plugin");
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
