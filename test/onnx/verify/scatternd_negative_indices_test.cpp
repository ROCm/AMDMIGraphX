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

// ScatterND allows negative indices, which count back from the end of their dimension. The
// indices are a runtime input here, so nothing can normalize them while parsing.
TEST_CASE(scatternd_negative_indices_test)
{
    migraphx::program p = read_onnx("scatternd_negative_indices_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape data_s{migraphx::shape::float_type, {1, 8, 4}};
    std::vector<float> data(32, 0.0f);
    migraphx::shape ind_s{migraphx::shape::int64_type, {1, 3, 2}};
    std::vector<int64_t> indices{0, 2, -1, -1, -1, 1};
    migraphx::shape upd_s{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> updates{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    migraphx::parameter_map pp;
    pp["data"]    = migraphx::argument(data_s, data.data());
    pp["indices"] = migraphx::argument(ind_s, indices.data());
    pp["updates"] = migraphx::argument(upd_s, updates.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{0, 0, 0, 0, 9, 10, 11, 12, 1, 2, 3, 4, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 5, 6, 7, 8};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
