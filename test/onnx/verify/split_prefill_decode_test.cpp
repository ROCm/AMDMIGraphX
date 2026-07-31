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
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

namespace {

std::vector<float>
run(migraphx::program& p, const std::vector<std::size_t>& lens, std::vector<float> data)
{
    migraphx::parameter_map params;
    params["x"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, lens}, data.data()};

    std::vector<float> result;
    p.eval(params).back().visit([&](auto output) { result.assign(output.begin(), output.end()); });
    return result;
}

} // namespace

// One compiled program serves both a single decode token and a full prefill sequence; the
// select_module picks the specialization from the shape of the input.
TEST_CASE(split_prefill_decode_test)
{
    migraphx::onnx_options options;
    options.split_prefill_decode          = true;
    options.use_symbolic_shapes           = true;
    options.dim_params["sequence_length"] = {1, 4};
    auto p                                = read_onnx("split_prefill_decode_test.onnx", options);
    p.compile(migraphx::make_target("ref"));

    EXPECT(migraphx::verify::verify_rms_range(run(p, {1, 1, 2}, {1.0f, 2.0f}),
                                              std::vector<float>{2.0f, 3.0f}));
    EXPECT(migraphx::verify::verify_rms_range(
        run(p, {1, 4, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}),
        std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}));
}
