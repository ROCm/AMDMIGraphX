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
#ifndef MIGRAPHX_GUARD_RTGLIB_REGISTER_OP_HPP
#define MIGRAPHX_GUARD_RTGLIB_REGISTER_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/auto_register.hpp>
#include <cstring>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void register_op(const operation& op);
operation load_op(const std::string& name);
bool has_op(const std::string& name);
std::vector<std::string> get_operators();

template <class T>
void register_op()
{
    register_op(T{});
}

struct register_op_action
{
    template <class T>
    static void apply()
    {
        register_op<T>();
    }
};

template <class T>
using auto_register_op = auto_register<register_op_action, T>;

#define MIGRAPHX_REGISTER_OP(...) MIGRAPHX_AUTO_REGISTER(register_op_action, __VA_ARGS__)

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
