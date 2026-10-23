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
#include <numeric>

namespace {

std::vector<migraphx::argument> eval_gqo(const std::string& name, std::size_t seq, bool with_bias)
{
    auto p = read_onnx(name);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape qkv_s{migraphx::shape::float_type, {1, seq, 48}};
    std::vector<float> qkv_data(qkv_s.elements());
    std::iota(qkv_data.begin(), qkv_data.end(), 0.0f);
    migraphx::literal qkv{qkv_s, qkv_data};

    migraphx::shape kv_dummy_s{migraphx::shape::float_type, {1}};
    migraphx::literal key{kv_dummy_s, std::vector<float>{0}};
    migraphx::literal value{kv_dummy_s, std::vector<float>{0}};

    migraphx::shape past_s{migraphx::shape::float_type, {1, 1, 4, 16}};
    std::vector<float> past_data(past_s.elements(), 0.25f);
    migraphx::literal past_key{past_s, past_data};
    migraphx::literal past_value{past_s, past_data};

    migraphx::shape b_s{migraphx::shape::uint8_type, {4, 1, 8}};
    std::vector<uint8_t> b_data{0x2,  0xe3, 0xc7, 0x89, 0xbd, 0xbe, 0x50, 0x41, 0xe9, 0xb4, 0xd4,
                                0x54, 0xc6, 0xb2, 0xfa, 0x27, 0x14, 0x3d, 0xbb, 0xe7, 0xa5, 0x0,
                                0x52, 0x28, 0xc1, 0xd9, 0x1f, 0x33, 0x16, 0x1e, 0x8b, 0x3c};
    migraphx::literal qweight{b_s, b_data};

    migraphx::shape scales_s{migraphx::shape::float_type, {4}};
    migraphx::literal scales{scales_s, std::vector<float>{1, 2, 3, 4}};

    migraphx::shape zp_s{migraphx::shape::uint8_type, {4}};
    migraphx::literal qzeros{zp_s, std::vector<uint8_t>{0x08, 0x09, 0x0a, 0x0b}};

    migraphx::shape bias_s{migraphx::shape::float_type, {4}};
    migraphx::literal bias{bias_s, std::vector<float>{1, 2, 3, 4}};

    migraphx::parameter_map pm;
    pm["qkv"]                   = qkv.get_argument();
    pm["key"]                   = key.get_argument();
    pm["value"]                 = value.get_argument();
    pm["past_key_values_key"]   = past_key.get_argument();
    pm["past_key_values_value"] = past_value.get_argument();
    pm["qweight"]               = qweight.get_argument();
    pm["scales"]                = scales.get_argument();
    pm["qzeros"]                = qzeros.get_argument();
    if(with_bias)
        pm["bias"] = bias.get_argument();
    return p.eval(pm);
}

std::vector<float> copy_floats(const migraphx::argument& arg)
{
    std::vector<float> out;
    arg.visit([&](auto output) { out.assign(output.begin(), output.end()); });
    return out;
}

void expect_matches_unfused(const std::string& fused,
                            const std::string& unfused,
                            std::size_t seq,
                            bool with_bias)
{
    auto fused_outs   = eval_gqo(fused, seq, with_bias);
    auto unfused_outs = eval_gqo(unfused, seq, with_bias);
    EXPECT(fused_outs.size() == 3);
    EXPECT(unfused_outs.size() == 3);
    for(std::size_t i = 0; i < fused_outs.size(); ++i)
    {
        EXPECT(fused_outs[i].get_shape().lens() == unfused_outs[i].get_shape().lens());
        EXPECT(migraphx::verify::allclose(copy_floats(fused_outs[i]),
                                          copy_floats(unfused_outs[i]),
                                          migraphx::verify::tolerance{}));
    }
}

} // namespace

TEST_CASE(gqo_decode_test)
{
    expect_matches_unfused("gqo_decode_test.onnx", "gqo_decode_unfused_test.onnx", 1, true);
}

TEST_CASE(gqo_prefill_test)
{
    expect_matches_unfused("gqo_prefill_test.onnx", "gqo_prefill_unfused_test.onnx", 2, true);
}

TEST_CASE(gqo_nobias_test)
{
    expect_matches_unfused("gqo_nobias_test.onnx", "gqo_nobias_unfused_test.onnx", 1, false);
}

TEST_CASE(gqo_head_sink_test)
{
    expect_matches_unfused("gqo_head_sink_test.onnx", "gqo_head_sink_unfused_test.onnx", 1, true);
}
