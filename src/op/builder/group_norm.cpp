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

// group_norm has no native op: normalize each channel group, then the affine.
struct group_norm : op_builder<group_norm>
{
    float epsilon      = 1e-5f;
    int64_t num_groups = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.epsilon, "epsilon"), f(self.num_groups, "num_groups"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x    = args[0];
        auto lens = x->get_shape().lens();
        if(lens.size() <= 2 or lens[1] % num_groups != 0)
            MIGRAPHX_THROW("group_norm op_builder: input rank must be > 2 and num_groups must "
                           "divide the channel dim");

        std::vector<int64_t> grouped_dims = {static_cast<int64_t>(lens[0]), num_groups, -1};
        auto grouped = m.insert_instruction(ins, make_op("reshape", {{"dims", grouped_dims}}), x);
        auto norm    = normalize(m, ins, grouped, {-1}, epsilon);

        std::vector<int64_t> out_dims(lens.begin(), lens.end());
        auto norm_r = m.insert_instruction(ins, make_op("reshape", {{"dims", out_dims}}), norm);

        // unsqueeze the per-channel scale/bias to broadcast over the spatial dims
        std::vector<int64_t> unsqueeze_axes(lens.size() - 2);
        std::iota(unsqueeze_axes.begin(), unsqueeze_axes.end(), 1);
        auto scale =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", unsqueeze_axes}}), args[1]);
        auto bias =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", unsqueeze_axes}}), args[2]);
        auto scaled = insert_common_op(m, ins, "mul", norm_r, scale);
        return {insert_common_op(m, ins, "add", scaled, bias)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
