/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <onnx_verify_utils.hpp>

TEST_CASE(skip_simplified_layer_normalization_test)
{
    using migraphx::half;
    std::vector<half> x{half{0.8},
                        half{-0.5},
                        half{0.0},
                        half{1.0},
                        half{0.5},
                        half{0.2},
                        half{0.3},
                        half{-0.6},
                        half{10.0},
                        half{-1.0},
                        half{0.0},
                        half{1.0},
                        half{1.2},
                        half{3.2},
                        half{-4.1},
                        half{5.3}};
    std::vector<half> skip{half{1.2},
                           half{-1.0},
                           half{2.0},
                           half{1.0},
                           half{1.5},
                           half{2.2},
                           half{-3.3},
                           half{2.6},
                           half{1.0},
                           half{-10.0},
                           half{-1.0},
                           half{2.0},
                           half{-1.2},
                           half{1.6},
                           half{-1.1},
                           half{1.3}};
    std::vector<half> scale{half{0.1}, half{0.2}, half{4.0}, half{-2.2}};

    auto p = read_onnx("skip_simplified_layer_normalization_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_x{migraphx::shape::half_type, {2, 2, 4}};
    migraphx::shape s_s{migraphx::shape::half_type, {4}};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, x.data());
    pp["skip"]  = migraphx::argument(s_x, skip.data());
    pp["gamma"] = migraphx::argument(s_s, scale.data());

    auto result = p.eval(pp).back();

    std::vector<half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<half> gold = {half{0.10596},
                              half{-0.1589},
                              half{4.24},
                              half{-2.33},
                              half{0.0838},
                              half{0.2012},
                              half{-5.03},
                              half{-1.844},
                              half{0.1385},
                              half{-0.277},
                              half{-0.504},
                              half{-0.831},
                              half{0.0},
                              half{0.1984},
                              half{-4.3},
                              half{-3.0}};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}