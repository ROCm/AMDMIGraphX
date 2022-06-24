/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/compiler.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

auto& compiler_map()
{
    static std::unordered_map<std::string, compiler_compile> m; // NOLINT
    return m;
}

auto& compiler_op_map()
{
    static std::unordered_map<std::string, compiler_compile_op> m; // NOLINT
    return m;
}

void register_compiler(const std::string& name, compiler_compile c, compiler_compile_op cop)
{
    compiler_map()[name]    = std::move(c);
    compiler_op_map()[name] = std::move(cop);
}

bool has_compiler_for(const std::string& name) { return compiler_map().count(name) > 0; }
compiler_replace compile(context& ctx, instruction_ref ins, const operation& op)
{
    return compiler_map().at(op.name())(ctx, ins, op);
}
operation
compile_op(const std::string& name, context& ctx, const std::vector<shape>& inputs, const value& v)
{
    return compiler_op_map().at(name)(ctx, inputs, v);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
