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

// Regression: indices arrive at scatternd with non-packed strides because the
// upstream Transpose only relabels the layout. Both the ref op and the GPU
// kernel must walk the indices tensor stride-aware -- the historical bug was
// that they read `k` contiguous memory positions per row and so swapped
// columns. Vision_encoder triggers this via unsqueeze->transpose->slice->
// concat. The matching test_scatternd_nonpacked_indices in test/verify/
// catches the GPU kernel; this one keeps the ref path honest end-to-end
// from the ONNX parser.
TEST_CASE(scatternd_nonpacked_indices_test)
{
    migraphx::program p = read_onnx("scatternd_nonpacked_indices_test.onnx");
    p.compile(migraphx::make_target("ref"));

    constexpr std::size_t n = 16;

    migraphx::shape data_s{migraphx::shape::float_type, {1, n}};
    std::vector<float> data(n, 0.0f);

    migraphx::shape upd_s{migraphx::shape::float_type, {n}};
    std::vector<float> updates(n);
    for(std::size_t i = 0; i < n; ++i)
        updates[i] = static_cast<float>(i + 1);

    migraphx::parameter_map pp;
    pp["data"]    = migraphx::argument(data_s, data.data());
    pp["updates"] = migraphx::argument(upd_s, updates.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold(n);
    for(std::size_t i = 0; i < n; ++i)
        gold[i] = static_cast<float>(n - i);

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
