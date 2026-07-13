/* The MIT License (MIT)
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

#include <numeric>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/op/builder/normalize.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// instance_norm has no native op: normalize from input stats, then the affine.
struct instance_norm : op_builder<instance_norm>
{
    float epsilon = 1e-5f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.epsilon, "epsilon"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x    = args[0];
        auto rank = static_cast<int64_t>(x->get_shape().ndim());
        if(rank < 2)
            MIGRAPHX_THROW("instance_norm op_builder: input rank must be at least 2");

        // reduce over the batch and spatial dims, keeping the channel dim
        std::vector<int64_t> axes = {0};
        for(int64_t i = 2; i < rank; ++i)
            axes.push_back(i);

        auto norm = normalize(m, ins, x, axes, epsilon);

        // unsqueeze the per-channel scale/bias to broadcast over the spatial dims
        auto scale = args[1];
        auto bias  = args[2];
        if(rank > 2)
        {
            std::vector<int64_t> unsqueeze_axes(rank - 2);
            std::iota(unsqueeze_axes.begin(), unsqueeze_axes.end(), 1);
            scale =
                m.insert_instruction(ins, make_op("unsqueeze", {{"axes", unsqueeze_axes}}), scale);
            bias =
                m.insert_instruction(ins, make_op("unsqueeze", {{"axes", unsqueeze_axes}}), bias);
        }
        auto scaled = insert_common_op(m, ins, "mul", norm, scale);
        return {insert_common_op(m, ins, "add", scaled, bias)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
