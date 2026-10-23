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

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

namespace {

instruction_ref unpack(module& m, instruction_ref ins, int n, int dim1, instruction_ref x)
{
    x = m.insert_instruction(ins, make_op("reshape", {{"dims", {n, -1}}}), x);
    x = m.insert_instruction(ins, make_op("unpack_int4"), x);
    if(x->get_shape().lens()[1] > dim1)
    {
        x = m.insert_instruction(
            ins, make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {dim1}}}), x);
    }
    return x;
}

instruction_ref prepare_blockwise_dq_arg(
    module& m, instruction_ref ins, int n, int k, int block_size, instruction_ref x)
{
    x = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {2}}}), x);

    auto bc_lens = x->get_shape().lens();
    bc_lens[2]   = block_size;
    x            = m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", bc_lens}}), x);
    x            = m.insert_instruction(ins, make_op("reshape", {{"dims", {n, -1}}}), x);

    if(x->get_shape().lens()[1] > k)
    {
        x = m.insert_instruction(
            ins, make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {k}}}), x);
    }

    return x;
}

instruction_ref matmul(module& m, instruction_ref ins, instruction_ref a, instruction_ref b)
{
    const auto a_rank = a->get_shape().ndim();
    if(a_rank == 1)
    {
        a = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0}}}), a);
    }
    else if(a_rank > 2)
    {
        auto b_lens    = b->get_shape().lens();
        auto b_bc_lens = a->get_shape().lens();
        std::copy(b_lens.begin(), b_lens.end(), b_bc_lens.end() - 2);
        b = m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", b_bc_lens}}), b);
    }

    auto dot = m.insert_instruction(ins, make_op("dot"), a, b);

    if(a_rank == 1)
        dot = m.insert_instruction(
            ins, make_op("squeeze", {{"axes", {dot->get_shape().ndim() - 2}}}), dot);

    return dot;
}

} // namespace

struct matmulnbits : op_builder<matmulnbits>
{
    int n          = 1;
    int k          = 1;
    int bits       = 4;
    int block_size = 32;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.n, "N"), f(self.k, "K"), f(self.bits, "bits"), f(self.block_size, "block_size"));
    }

    // args: A, B, scales, optional zero_points, optional bias
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        if(args.size() < 3)
            MIGRAPHX_THROW("MatMulNBits: expects at least 3 inputs (A, B, scales), got " +
                           std::to_string(args.size()));

        auto a              = args[0];
        auto b              = args[1];
        auto scales         = args[2];
        const bool has_zp   = args.size() > 3 and not args[3]->is_undefined();
        const bool has_bias = args.size() > 4 and not args[4]->is_undefined();

        if(bits != 4)
            MIGRAPHX_THROW("MatMulNBits: bits only supported for value of 4, actual value " +
                           std::to_string(bits));

        if(block_size < 16 or (block_size & (block_size - 1)) != 0)
            MIGRAPHX_THROW("MatMulNBits: block_size must be a power of 2 and >=16, actual value " +
                           std::to_string(block_size));

        const int n_blocks_per_col = (k + block_size - 1) / block_size;
        const int blob_size        = (block_size * bits + 7) / 8;

        std::vector<std::size_t> expected_b_lens(3);
        expected_b_lens[0] = n;
        expected_b_lens[1] = n_blocks_per_col;
        expected_b_lens[2] = blob_size;
        if(b->get_shape().lens() != expected_b_lens)
            MIGRAPHX_THROW("MatMulNBits: Input B does not match expected dims: " +
                           to_string_range(expected_b_lens) +
                           ". Actual dims: " + to_string_range(b->get_shape().lens()));

        std::size_t expected_scales_lens = n;
        expected_scales_lens *= n_blocks_per_col;
        if(scales->get_shape().elements() != expected_scales_lens)
            MIGRAPHX_THROW("MatMulNBits: Input scales does not match expected dims: " +
                           to_string(expected_scales_lens) +
                           ". Actual dims: " + to_string_range(scales->get_shape().lens()));

        if(has_zp)
        {
            std::size_t expected_zp_elements = n;
            expected_zp_elements *= (n_blocks_per_col * bits + 7) / 8;
            if(args[3]->get_shape().elements() != expected_zp_elements)
                MIGRAPHX_THROW("MatMulNBits: Input zero_points does not match expected dims: " +
                               to_string(expected_zp_elements) +
                               ". Actual dims: " + to_string_range(args[3]->get_shape().lens()));
        }

        auto dq_b = unpack(m, ins, n, k, b);

        auto dq_scales = m.insert_instruction(ins, make_op("reshape", {{"dims", {n, -1}}}), scales);
        dq_scales      = prepare_blockwise_dq_arg(m, ins, n, k, block_size, dq_scales);

        instruction_ref dq_zp;
        if(has_zp)
        {
            dq_zp = unpack(m, ins, n, n_blocks_per_col, args[3]);
            dq_zp = prepare_blockwise_dq_arg(m, ins, n, k, block_size, dq_zp);
        }
        else
        {
            dq_zp = m.insert_literal(ins, literal{shape{shape::uint8_type, {1}}, {8}});
            dq_zp = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", dq_b->get_shape().lens()}}), dq_zp);
        }

        dq_b     = m.insert_instruction(ins, make_op("dequantizelinear"), dq_b, dq_scales, dq_zp);
        dq_b     = m.insert_instruction(ins, make_op("transpose", {{"permutation", {1, 0}}}), dq_b);
        auto out = matmul(m, ins, a, dq_b);

        if(has_bias)
            out = insert_common_op(m, ins, "add", out, args[4]);

        return {out};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
