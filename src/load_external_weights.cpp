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
#include <migraphx/ranges.hpp>
#include <migraphx/op/external_weight.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void load_external_weights::apply(module& m) const
{
    auto weight_refs =
        find_all(iterator_for(m), [](auto ins) { return ins->name() == "external_weight"; });

    for(auto ins : weight_refs)
    {
        const auto w = any_cast<op::external_weight>(ins->get_operator());

        auto raw = read_buffer(fs::path{base_dir} / w.location, w.offset, w.length);

        // The shape comes from the producer op and is always standard; the file region,
        // however, is external input, so its byte size is validated rather than asserted.
        assert(w.s.standard());
        if(raw.size() != w.s.bytes())
            MIGRAPHX_THROW("LOAD_EXTERNAL_WEIGHTS: weight \"" + w.location + "\" file size " +
                           std::to_string(raw.size()) + " does not match expected size " +
                           std::to_string(w.s.bytes()));

        // Insert the literal where the external_weight op sits so the baked program keeps
        // the same instruction order as a normal literal-based parse.
        auto lit = m.insert_literal(ins, literal{w.s, raw.data()});
        m.replace_instruction(ins, lit);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
