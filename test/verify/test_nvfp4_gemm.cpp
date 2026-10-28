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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

namespace {
// NVFP4 two-level scales as parsed from ONNX: fp8e4m3fn block scales dequantized by a
// per-tensor scale, then expanded by block_size along the last axis
migraphx::instruction_ref add_nvfp4_scale(migraphx::module_ref m,
                                          const std::vector<std::size_t>& lens,
                                          float per_tensor_scale,
                                          std::size_t block_size)
{
    using migraphx::make_op;
    // Block scales from 1 to 1.75, which fp8e4m3fn represents exactly
    std::vector<float> data(migraphx::shape{migraphx::shape::float_type, lens}.elements());
    for(std::size_t i = 0; i < data.size(); ++i)
        data[i] = 1.0f + 0.25f * float(i % 4);
    auto scale  = m->add_literal(migraphx::literal{{migraphx::shape::fp8e4m3fn_type, lens}, data});
    auto global = m->add_instruction(make_op("multibroadcast", {{"out_lens", lens}}),
                                     m->add_literal(per_tensor_scale));
    auto dq     = m->add_instruction(make_op("dequantizelinear"), scale, global);
    auto blocks = lens;
    blocks.push_back(block_size);
    auto unsqueeze = m->add_instruction(make_op("unsqueeze", {{"axes", {lens.size()}}}), dq);
    auto mbcast = m->add_instruction(make_op("multibroadcast", {{"out_lens", blocks}}), unsqueeze);
    auto expanded_lens = lens;
    expanded_lens.back() *= block_size;
    return m->add_instruction(make_op("reshape", {{"dims", expanded_lens}}), mbcast);
}

// Quantize to fp4 and dequantize again, as parsed from ONNX
migraphx::instruction_ref add_fp4_qdq(migraphx::module_ref m,
                                      migraphx::instruction_ref input,
                                      migraphx::instruction_ref scale)
{
    using migraphx::make_op;
    auto q = m->add_instruction(
        make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}), input, scale);
    q = m->add_instruction(make_op("pack_fp4"), q);
    q = m->add_instruction(make_op("unpack_fp4"), q);
    return m->add_instruction(make_op("dequantizelinear"), q, scale);
}
} // namespace

/**
 * NVFP4 GEMM: FP4 operands with fp8e4m3fn block scales (block size 16 along K) and a float
 * per-tensor scale. Weights are stored {N, K} and transposed, as exported from Linear layers.
 * gfx1250 runs it as a scaled GEMM, other GPUs dequantize.
 */
struct test_nvfp4_gemm : verify_program<test_nvfp4_gemm>
{
    migraphx::program create_program() const
    {
        const std::size_t m          = 32;
        const std::size_t k          = 256;
        const std::size_t n          = 64;
        const std::size_t block_size = 16;
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto input = mm->add_parameter("x1", migraphx::shape{migraphx::shape::float_type, {m, k}});
        // The inputs are within [-1, 1], so the scales keep them within the fp4 range
        input =
            add_fp4_qdq(mm, input, add_nvfp4_scale(mm, {m, k / block_size}, 0.1875f, block_size));
        auto weights = mm->add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {n, k}}, 2));
        weights =
            add_fp4_qdq(mm, weights, add_nvfp4_scale(mm, {n, k / block_size}, 0.25f, block_size));
        weights =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), weights);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), input, weights);
        mm->add_return({dot});
        return p;
    }
    std::string section() const { return "gemm"; }

    // Other GPUs remove the fp4 fake quantization of the input
    std::size_t get_tolerance() const { return 4e5; };
};
