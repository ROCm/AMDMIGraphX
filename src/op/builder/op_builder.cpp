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

#include <unordered_map>
#include <migraphx/ranges.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

std::unordered_map<std::string, builder_func>& builder_map()
{
    static std::unordered_map<std::string, builder_func> m; // NOLINT
    return m;
}

void register_builder(const std::string& name, builder_func f)
{
    builder_map()[name] = std::move(f);
}

std::vector<instruction_ref> insert(const std::string& name,
                                    module& m,
                                    instruction_ref ins,
                                    const std::vector<instruction_ref>& args,
                                    const value& options)
{
    return at(builder_map(), name, "Builder not found: " + name)(m, ins, args, options);
}

std::vector<instruction_ref> add(const std::string& name,
                                 module& m,
                                 const std::vector<instruction_ref>& args,
                                 const value& options)
{
    return at(builder_map(), name, "Builder not found: " + name)(m, m.end(), args, options);
}

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
