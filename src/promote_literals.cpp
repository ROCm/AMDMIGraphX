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

#include <migraphx/promote_literals.hpp>
#include <migraphx/hash.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <algorithm>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static std::size_t literal_hash(const literal& l)
{
    const auto& s    = l.get_shape();
    std::size_t seed = std::hash<std::string_view>{}(std::string_view{l.data(), s.bytes()});
    hash_combine(seed, s.type());
    hash_range(seed, s.lens().begin(), s.lens().end());
    hash_range(seed, s.strides().begin(), s.strides().end());
    return seed;
}

// Only literals are merged: the root module is already lowered here, so a general common
// subexpression pass would also merge identical allocations and alias unrelated buffers.
static void merge_duplicate_literals(module& m)
{
    std::unordered_map<std::size_t, std::vector<instruction_ref>> literals;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "@literal")
            continue;
        const auto& l   = ins->get_literal();
        auto& same_hash = literals[literal_hash(l)];
        auto existing   = std::find_if(same_hash.begin(), same_hash.end(), [&](instruction_ref x) {
            return x->get_literal() == l;
        });
        if(existing == same_hash.end())
            same_hash.push_back(ins);
        else
            m.replace_instruction(ins, *existing);
    }
}

void promote_literals::apply(module_pass_manager& mpm) const
{
    module& m              = mpm.get_module();
    module_ref root_module = mpm.get_root_module();
    if(m == *root_module)
    {
        // The root is visited last, after literals from every submodule have been promoted.
        merge_duplicate_literals(m);
        return;
    }

    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@literal")
        {
            auto new_lit     = root_module->add_literal(ins->get_literal());
            auto ins_outputs = ins->outputs();
            for(auto out_ins : ins_outputs)
            {
                migraphx::instruction::replace_argument(out_ins, ins, new_lit);
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
