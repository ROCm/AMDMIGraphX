/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_OP_BUILDER_REGISTER_BUILDER_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_OP_BUILDER_REGISTER_BUILDER_HPP

#include <functional>
#include <string>
#include <vector>
#include <migraphx/auto_register.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/module.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

using builder_func =
    std::function<std::vector<instruction_ref>(module& m,
                                               instruction_ref ins,
                                               const std::vector<instruction_ref>& args,
                                               const value& options)>;

MIGRAPHX_EXPORT void register_builder(const std::string& name, builder_func f);

template <class T>
void register_builder()
{
    builder_func f = [](module& m,
                        instruction_ref ins,
                        const std::vector<instruction_ref>& args,
                        const value& options) {
        auto x = from_value<T>(options);
        return x.insert(m, ins, args);
    };
    register_builder(T::name(), std::move(f));
}

struct register_builder_action
{
    template <class T>
    static void apply()
    {
        register_builder<T>();
    }
};

template <class T>
using op_builder = auto_register<register_builder_action, T>;

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
