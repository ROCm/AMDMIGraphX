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
#include <migraphx/promote_storage_type.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/eliminate_convert.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/replace_data_type.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static bool is_computation(instruction_ref ins)
{
    // convert and bit_cast are storage boundaries themselves, and layout and
    // identity don't compute anything even though they all carry the pointwise
    // attribute
    if(contains({"convert", "bit_cast", "layout", "identity"}, ins->name()))
        return false;
    auto attrs = ins->get_operator().attributes();
    return attrs.get("reduce", false) or attrs.get("pointwise", false);
}

void promote_storage_type::apply(module_pass_manager& mpm) const
{
    auto& m      = mpm.get_module();
    auto promote = [&](instruction_ref ins) {
        return contains(types, ins->get_shape().type()) and is_computation(ins);
    };
    if(none_of(iterator_for(m), promote))
        return;
    replace_data_type(m, types, shape::float_type, promote);
    // Adjacent promoted instructions are connected by a convert to the
    // storage type followed by a convert back to float, which cancel
    mpm.run_pass(eliminate_convert{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
