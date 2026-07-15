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

#include <cstddef>
#include <cstdint>
#include <vector>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// as_strided has no native op: gather each element from its strided storage offset.
struct torch_as_strided : op_builder<torch_as_strided>
{
    std::vector<int64_t> size;
    std::vector<int64_t> stride;
    int64_t storage_offset = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"),
                    f(self.stride, "stride"),
                    f(self.storage_offset, "storage_offset"));
    }

    static std::vector<std::string> names() { return {"tm::as_strided"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        if(size.size() != stride.size())
            MIGRAPHX_THROW("as_strided: size and stride must have the same length");
        shape strided{shape::int64_type,
                      std::vector<std::size_t>(size.begin(), size.end()),
                      std::vector<std::size_t>(stride.begin(), stride.end())};
        std::vector<int64_t> data(strided.elements());
        for(std::size_t i = 0; i < data.size(); ++i)
            data[i] = storage_offset + strided.index(i);
        auto indices = m.add_literal(
            literal{shape{shape::int64_type, {data.size()}}, data.begin(), data.end()});

        auto flat_inp = m.insert_instruction(ins, make_op("contiguous"), args[0]);
        flat_inp = m.insert_instruction(ins, make_op("reshape", {{"dims", {-1}}}), flat_inp);
        auto gathered = m.insert_instruction(ins, make_op("gather", {{"axis", 0}}), flat_inp, indices);
        return {m.insert_instruction(ins, make_op("reshape", {{"dims", size}}), gathered)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
