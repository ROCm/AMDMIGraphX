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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_where : op_parser<parse_where>
{
    std::vector<op_desc> operators() const { return {{"Where"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        // Fast path: when all three inputs already share the same dims, emit
        // where() directly (no redundant broadcast). This keeps the common
        // same-shape case -- including all-dynamic identical shapes -- as a
        // bare where op.
        auto dims_match = [](const shape& a, const shape& b) {
            if(a.dynamic() or b.dynamic())
                return a.dynamic() and b.dynamic() and a.dyn_dims() == b.dyn_dims();
            return a.lens() == b.lens();
        };
        const auto s0 = args[0]->get_shape();
        if(dims_match(args[1]->get_shape(), s0) and dims_match(args[2]->get_shape(), s0))
            return info.add_instruction(make_op("where"), args[0], args[1], args[2]);

        // Otherwise broadcast the three inputs to a common shape. where()
        // must NOT unify element types (args[0] is the bool condition while
        // args[1]/args[2] carry the data type), so add_common_op is called
        // with common_type=false. This handles static, dynamic, and mixed
        // inputs uniformly; the dynamic path goes through
        // compute_common_dyn_dims, so an unconstrained input (e.g. a
        // broadcast_with_dims / ONNX Expand output {0, SIZE_MAX}) is
        // intersected with the other operands instead of requiring every
        // input to already share an identical dynamic shape.
        return migraphx::add_common_op(
            *info.mod, make_op("where"), args, {/*common_type=*/false, /*common_lens=*/true});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
