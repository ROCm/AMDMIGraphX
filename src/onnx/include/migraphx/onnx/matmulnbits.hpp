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
#ifndef MIGRAPHX_GUARD_ONNX_MATMULNBITS_HPP
#define MIGRAPHX_GUARD_ONNX_MATMULNBITS_HPP

#include <migraphx/config.hpp>
#include <migraphx/onnx/onnx_parser.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>
#include <cmath>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {
namespace matmulnbits {

inline instruction_ref unpack(onnx_parser::node_info& info, int n, int dim1, instruction_ref x)
{
    x = info.add_instruction(make_op("reshape", {{"dims", {n, -1}}}), x);
    x = info.add_instruction(make_op("unpack_int4"), x);
    if(x->get_shape().lens()[1] > static_cast<std::size_t>(dim1))
    {
        x = info.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {dim1}}}), x);
    }
    return x;
}

inline instruction_ref prepare_blockwise_dq_arg(
    onnx_parser::node_info& info, int n, int k, int block_size, instruction_ref x)
{
    x = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), x);

    auto bc_lens = x->get_shape().lens();
    bc_lens[2]   = block_size;
    x            = info.add_instruction(make_op("multibroadcast", {{"out_lens", bc_lens}}), x);
    x            = info.add_instruction(make_op("reshape", {{"dims", {n, -1}}}), x);

    if(x->get_shape().lens()[1] > static_cast<std::size_t>(k))
    {
        x = info.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {k}}}), x);
    }

    return x;
}

inline instruction_ref matmul(onnx_parser::node_info& info, instruction_ref a, instruction_ref b)
{
    const auto a_rank = a->get_shape().ndim();
    if(a_rank == 1)
    {
        a = info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), a);
    }
    else if(a_rank > 2)
    {
        auto b_lens    = b->get_shape().lens();
        auto b_bc_lens = a->get_shape().lens();
        std::copy(b_lens.begin(), b_lens.end(), b_bc_lens.end() - 2);
        b = info.add_instruction(make_op("multibroadcast", {{"out_lens", b_bc_lens}}), b);
    }

    auto dot = info.add_instruction(make_op("dot"), a, b);

    if(a_rank == 1)
        dot = info.add_instruction(
            make_op("squeeze", {{"axes", {dot->get_shape().ndim() - 2}}}), dot);

    return dot;
}

// Lower A @ dequant(B) [+ bias].
inline instruction_ref add_matmul_nbits(onnx_parser::node_info& info,
                                        instruction_ref a,
                                        instruction_ref b,
                                        instruction_ref scales,
                                        std::optional<instruction_ref> zp,
                                        std::optional<instruction_ref> bias,
                                        int n,
                                        int k,
                                        int bits,
                                        int block_size)
{
    if(bits != 4)
        MIGRAPHX_THROW("MatMulNBits: bits only supported for value of 4, actual value " +
                       std::to_string(bits));

    if(block_size < 16 or (block_size & (block_size - 1)) != 0)
        MIGRAPHX_THROW("MatMulNBits: block_size must be a power of 2 and >=16, actual value " +
                       std::to_string(block_size));

    const size_t n_blocks_per_col = (k + block_size - 1) / block_size;
    const size_t blob_size        = std::ceil(block_size * bits / 8.0f);

    std::vector<size_t> expected_b_lens{static_cast<size_t>(n), n_blocks_per_col, blob_size};
    if(b->get_shape().lens() != expected_b_lens)
        MIGRAPHX_THROW("MatMulNBits: Input B does not match expected dims: " +
                       to_string_range(expected_b_lens) +
                       ". Actual dims: " + to_string_range(b->get_shape().lens()));

    const size_t expected_scales_lens = static_cast<size_t>(n) * n_blocks_per_col;
    if(scales->get_shape().elements() != expected_scales_lens)
        MIGRAPHX_THROW("MatMulNBits: Input scales does not match expected dims: " +
                       to_string(expected_scales_lens) +
                       ". Actual dims: " + to_string_range(scales->get_shape().lens()));

    const bool has_zp = zp.has_value() and not (*zp)->is_undefined();
    if(has_zp)
    {
        const size_t expected_zp_elements =
            static_cast<size_t>(n) * static_cast<size_t>(std::ceil(n_blocks_per_col * bits / 8.0f));
        if((*zp)->get_shape().elements() != expected_zp_elements)
            MIGRAPHX_THROW("MatMulNBits: Input zero_points does not match expected dims: " +
                           to_string(expected_zp_elements) +
                           ". Actual dims: " + to_string_range((*zp)->get_shape().lens()));
    }

    auto dq_b = unpack(info, n, k, b);

    auto dq_scales = info.add_instruction(make_op("reshape", {{"dims", {n, -1}}}), scales);
    dq_scales      = prepare_blockwise_dq_arg(info, n, k, block_size, dq_scales);

    instruction_ref dq_zp;
    if(has_zp)
    {
        dq_zp = unpack(info, n, static_cast<int>(n_blocks_per_col), *zp);
        dq_zp = prepare_blockwise_dq_arg(info, n, k, block_size, dq_zp);
    }
    else
    {
        dq_zp = info.add_literal(literal{shape{shape::uint8_type, {1}}, {8}});
        dq_zp = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", dq_b->get_shape().lens()}}), dq_zp);
    }

    dq_b = info.add_instruction(make_op("dequantizelinear"), {dq_b, dq_scales, dq_zp});
    dq_b = info.add_instruction(make_op("transpose", {{"permutation", {1, 0}}}), dq_b);
    auto out = matmul(info, a, dq_b);

    if(bias.has_value() and not (*bias)->is_undefined())
        out = info.add_common_op("add", out, *bias);

    return out;
}

} // namespace matmulnbits
} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
