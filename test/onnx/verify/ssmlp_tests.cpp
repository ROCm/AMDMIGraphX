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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(ssmlp_fp16_test)
{
    auto p = read_onnx("ssmlp_fp16_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto half = migraphx::shape::half_type;
    migraphx::shape row{half, {1, 16}};
    migraphx::shape vec{half, {16}};
    migraphx::shape w_s{migraphx::shape::uint8_type, {16, 1, 8}};
    migraphx::shape z_s{migraphx::shape::uint8_type, {16}};

    // residual 100 + attn 200 = 300. With zero weights the MLP output is 0, so both
    // normalizations see a row of 300. gamma 2 then 3 => normalized outputs 2 and 3.
    // The returned residual is the second skip sum, which stays 300.
    migraphx::literal residual{row, std::vector<float>(16, 100.f)};
    migraphx::literal attn{row, std::vector<float>(16, 200.f)};
    migraphx::literal gamma1{vec, std::vector<float>(16, 2.f)};
    migraphx::literal gamma2{vec, std::vector<float>(16, 3.f)};
    migraphx::literal zeros_w{w_s, std::vector<uint8_t>(w_s.elements(), 0)};
    migraphx::literal ones_s{vec, std::vector<float>(16, 1.f)};
    migraphx::literal zeros_z{z_s, std::vector<uint8_t>(z_s.elements(), 0)};

    migraphx::parameter_map pm;
    pm["residual"] = residual.get_argument();
    pm["attn"]     = attn.get_argument();
    pm["gamma1"]   = gamma1.get_argument();
    pm["gamma2"]   = gamma2.get_argument();
    pm["gate_b"]   = zeros_w.get_argument();
    pm["gate_s"]   = ones_s.get_argument();
    pm["gate_z"]   = zeros_z.get_argument();
    pm["up_b"]     = zeros_w.get_argument();
    pm["up_s"]     = ones_s.get_argument();
    pm["up_z"]     = zeros_z.get_argument();
    pm["down_b"]   = zeros_w.get_argument();
    pm["down_s"]   = ones_s.get_argument();
    pm["down_z"]   = zeros_z.get_argument();

    auto outs = p.eval(pm);
    EXPECT(outs.size() == 2);

    std::vector<float> norm;
    std::vector<float> skip;
    outs[0].visit([&](auto output) { norm.assign(output.begin(), output.end()); });
    outs[1].visit([&](auto output) { skip.assign(output.begin(), output.end()); });

    EXPECT(outs[0].get_shape().lens() == std::vector<std::size_t>{1, 16});
    EXPECT(outs[1].get_shape().lens() == std::vector<std::size_t>{1, 16});
    // fp16 rsqrt of a clipped mean square would land near 3.5 instead of 3.
    EXPECT(migraphx::verify::allclose(norm, std::vector<float>(16, 3.f), {0.05, 0.05, 0.05}));
    EXPECT(migraphx::verify::allclose(skip, std::vector<float>(16, 300.f), {0.05, 0.05, 0.05}));
}
