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

#include <cstdint>
#include <vector>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/op/builder/op_builder.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// index_copy has no native op: scatter src into the rows of `dim` listed in the 1-D index.
struct torch_index_copy : op_builder<torch_index_copy>
{
    int64_t dim = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dim, "dim"));
    }

    static std::vector<std::string> names() { return {"tm::index_copy"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto inp      = args[0];
        auto idx      = args[1];
        auto src      = args[2];
        auto src_lens = src->get_shape().lens();
        auto axis     = tune_axis(src_lens.size(), dim, "index_copy");

        std::vector<int64_t> rsp(src_lens.size(), 1);
        rsp[axis]        = idx->get_shape().lens().at(0);
        auto scatter_idx = m.insert_instruction(ins, make_op("reshape", {{"dims", rsp}}), idx);
        scatter_idx      = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", src_lens}}), scatter_idx);
        return {m.insert_instruction(
            ins, make_op("scatter_none", {{"axis", axis}}), {inp, scatter_idx, src})};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
