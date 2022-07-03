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
#ifndef MIGRAPHX_GUARD_GPU_COMPILER_HPP
#define MIGRAPHX_GUARD_GPU_COMPILER_HPP

#include <migraphx/config.hpp>
#include <migraphx/auto_register.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/value.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

using compiler_replace = std::function<void(module& m, instruction_ref ins)>;
using compiler_compile = std::function<compiler_replace(context&, instruction_ref, operation)>;
using compiler_compile_op =
    std::function<operation(context&, const std::vector<shape>& inputs, const value&)>;

void register_compiler(const std::string& name, compiler_compile c, compiler_compile_op cop);

bool has_compiler_for(const std::string& name);
compiler_replace compile(context& ctx, instruction_ref ins, const operation& op);
operation
compile_op(const std::string& name, context& ctx, const std::vector<shape>& inputs, const value& v);

template <class T>
void register_compiler()
{
    T c;
    for(auto&& name : c.names())
    {
        register_compiler(
            name,
            [=](auto&&... xs) { return c.compile(std::forward<decltype(xs)>(xs)...); },
            [=](auto&&... xs) { return c.compile_op(std::forward<decltype(xs)>(xs)...); });
    }
}

struct register_compiler_action
{
    template <class T>
    static void apply()
    {
        register_compiler<T>();
    }
};

template <class T>
using auto_register_compiler = auto_register<register_compiler_action, T>;

template <class Derived>
struct compiler : auto_register_compiler<Derived>
{
    auto replace(const operation& op) const
    {
        return
            [=](module& m, instruction_ref ins) { m.replace_instruction(ins, op, ins->inputs()); };
    }
    operation compile_op(context&, const std::vector<shape>&, const value&) const { return {}; }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_COMPILER_HPP
