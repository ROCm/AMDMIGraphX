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

#include <onnx_test.hpp>

namespace {

void expect_same_program(const std::string& fused, const std::string& unfused)
{
    auto fused_prog   = optimize_onnx(fused);
    auto unfused_prog = optimize_onnx(unfused);
    fused_prog.sort();
    unfused_prog.sort();
    EXPECT(fused_prog == unfused_prog);
}

} // namespace

TEST_CASE(gqo_decode_test)
{
    expect_same_program("gqo_decode_test.onnx", "gqo_decode_unfused_test.onnx");
}

TEST_CASE(gqo_prefill_test)
{
    expect_same_program("gqo_prefill_test.onnx", "gqo_prefill_unfused_test.onnx");
}

TEST_CASE(gqo_nobias_test)
{
    expect_same_program("gqo_nobias_test.onnx", "gqo_nobias_unfused_test.onnx");
}

TEST_CASE(gqo_head_sink_test)
{
    expect_same_program("gqo_head_sink_test.onnx", "gqo_head_sink_unfused_test.onnx");
}

TEST_CASE(gqo_position_ids_test)
{
    EXPECT(test::throws([&] { read_onnx("gqo_position_ids_test.onnx"); }));
}
