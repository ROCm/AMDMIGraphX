/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/eliminate_device_copy.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

struct find_redundant_copy : match::supports_dynamic_shapes
{
    auto matcher() const
    {
        return match::name("hip::copy_from_gpu")(match::arg(0)(match::name("hip::copy_to_gpu")));
    }

    static bool is_returned(const module& m, instruction_ref copy)
    {
        auto returns = m.get_returns();
        return std::find(returns.begin(), returns.end(), copy) != returns.end();
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto copy        = r.result;
        auto source_copy = copy->inputs().front();
        auto source      = source_copy->inputs().front();
        if(not is_returned(m, copy) and copy->get_shape() == source->get_shape())
            m.replace_instruction(copy, source);
    }
};

} // namespace

void eliminate_device_copy::apply(module& m) const
{
    match::find_matches(m, find_redundant_copy{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
