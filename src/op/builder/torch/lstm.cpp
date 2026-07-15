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
#include <vector>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// lstm expands into an lstm op plus its last hidden-state and last cell-state outputs.
struct torch_lstm : op_builder<torch_lstm>
{
    std::size_t hidden_size = 1;
    std::vector<operation> actv_funcs{};
    rnn_direction direction = rnn_direction::forward;
    float clip              = 0.0f;
    int input_forget        = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.hidden_size, "hidden_size"),
                    f(self.actv_funcs, "actv_func"),
                    f(self.direction, "direction"),
                    f(self.clip, "clip"),
                    f(self.input_forget, "input_forget"));
    }

    static std::vector<std::string> names() { return {"tm::lstm"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto self = *this;
        if(self.actv_funcs.empty())
        {
            self.actv_funcs = {make_op("sigmoid"), make_op("tanh"), make_op("tanh")};
            if(self.direction == rnn_direction::bidirectional)
            {
                self.actv_funcs.insert(self.actv_funcs.end(),
                                       {make_op("sigmoid"), make_op("tanh"), make_op("tanh")});
            }
        }
        auto hidden_states =
            m.insert_instruction(ins, make_op("lstm", migraphx::to_value(self)), args);
        auto last_hs   = m.insert_instruction(ins, make_op("rnn_last_hs_output"), hidden_states);
        auto last_cell = m.insert_instruction(ins, make_op("rnn_last_cell_output"), hidden_states);
        return {hidden_states, last_hs, last_cell};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
