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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// conv_transpose has no native op: convolution_backwards + output_padding crop + channel bias.
struct torch_conv_transpose : op_builder<torch_conv_transpose>
{
    std::vector<std::size_t> stride;
    std::vector<std::size_t> padding;
    std::vector<std::size_t> dilation;
    std::vector<std::size_t> output_padding;
    int group = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.stride, "stride"),
                    f(self.padding, "padding"),
                    f(self.dilation, "dilation"),
                    f(self.output_padding, "output_padding"),
                    f(self.group, "group"));
    }

    static std::vector<std::string> names() { return {"tm::conv_transpose"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        // output_padding cannot be expressed by the op: run it unpadded, then crop
        bool crop = std::any_of(
            output_padding.begin(), output_padding.end(), [](std::size_t o) { return o != 0; });
        auto pad = crop ? std::vector<std::size_t>(padding.size(), 0) : padding;
        auto out = m.insert_instruction(ins,
                                        make_op("convolution_backwards",
                                                {{"stride", stride},
                                                 {"padding", pad},
                                                 {"dilation", dilation},
                                                 {"group", group}}),
                                        args[0],
                                        args[1]);

        if(crop)
        {
            auto spatial = out->get_shape().lens();
            std::vector<int64_t> axes(output_padding.size());
            std::vector<int64_t> starts(output_padding.size());
            std::vector<int64_t> ends(output_padding.size());
            for(std::size_t i = 0; i < output_padding.size(); ++i)
            {
                axes[i]   = static_cast<int64_t>(i + 2);
                starts[i] = static_cast<int64_t>(padding[i]);
                ends[i]   = static_cast<int64_t>(spatial[i + 2] - padding[i] + output_padding[i]);
            }
            out = m.insert_instruction(
                ins, make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), out);
        }

        if(args.size() < 3)
            return {out};

        auto out_lens = out->get_shape().lens();
        auto bias     = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", out_lens}}), args[2]);
        return {m.insert_instruction(ins, make_op("add"), out, bias)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
