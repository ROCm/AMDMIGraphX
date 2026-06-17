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
#include <migraphx/load_external_weights.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void load_external_weights::apply(module& m) const
{
    std::vector<instruction_ref> weight_refs;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "external_weight")
            weight_refs.push_back(ins);
    }

    for(auto ins : weight_refs)
    {
        auto v              = ins->get_operator().to_value();
        const auto location = v.at("location").to<std::string>();
        const auto offset   = v.at("offset").to<std::size_t>();
        const auto length   = v.at("length").to<std::size_t>();
        const auto s        = ins->get_shape();

        auto raw = read_buffer(fs::path{base_dir} / location, offset, length);

        // The flat on-disk buffer is copied element-for-element into the literal, so
        // the weight must have a standard layout whose byte size matches the file region.
        if(not s.standard())
            MIGRAPHX_THROW("LOAD_EXTERNAL_WEIGHTS: weight \"" + location +
                           "\" does not have a standard shape");
        if(raw.size() != s.bytes())
            MIGRAPHX_THROW("LOAD_EXTERNAL_WEIGHTS: weight \"" + location + "\" file size " +
                           std::to_string(raw.size()) + " does not match expected size " +
                           std::to_string(s.bytes()));

        auto lit = m.add_literal(literal{s, raw.data()});
        m.replace_instruction(ins, lit);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
