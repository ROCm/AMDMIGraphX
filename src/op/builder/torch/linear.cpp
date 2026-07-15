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
#include <migraphx/value.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/op/builder/op_builder.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// linear reuses the gemm builder; ND inputs are flattened to rank 2.
struct torch_linear : op_builder<torch_linear>
{
    static std::vector<std::string> names() { return {"tm::linear"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        const value gemm_opts{{"transB", true}};
        auto lens = args[0]->get_shape().lens();
        if(lens.size() == 2)
            return op::builder::insert("gemm", m, ins, args, gemm_opts);

        auto rows                 = args[0]->get_shape().elements() / lens.back();
        std::vector<int64_t> flat = {static_cast<int64_t>(rows),
                                     static_cast<int64_t>(lens.back())};
        auto x2d = m.insert_instruction(ins, make_op("reshape", {{"dims", flat}}), args[0]);

        auto gemm_args = args;
        gemm_args[0]   = x2d;
        auto out       = op::builder::insert("gemm", m, ins, gemm_args, gemm_opts).front();

        std::vector<int64_t> out_dims(lens.begin(), lens.end() - 1);
        out_dims.push_back(static_cast<int64_t>(out->get_shape().lens().back()));
        return {m.insert_instruction(ins, make_op("reshape", {{"dims", out_dims}}), out)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
