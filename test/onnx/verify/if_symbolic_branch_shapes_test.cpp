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

// The branches return outputs of different lengths; the one taken at run time gives its shape
TEST_CASE(if_symbolic_branch_shapes_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes = true;
    options.dim_params          = {{"N", migraphx::shape::dynamic_dimension{1, 8}},
                                   {"M", migraphx::shape::dynamic_dimension{1, 8}}};
    migraphx::program p         = read_onnx("if_symbolic_branch_shapes_test.onnx", options);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape xs{migraphx::shape::float_type, {3, 4}};
    migraphx::shape ys{migraphx::shape::float_type, {5, 4}};
    std::vector<float> x(xs.elements());
    std::vector<float> y(ys.elements());
    std::iota(x.begin(), x.end(), 0.0f);
    std::iota(y.begin(), y.end(), 100.0f);

    for(bool cond : {true, false})
    {
        migraphx::parameter_map pp;
        pp["x"]      = migraphx::argument(xs, x.data());
        pp["y"]      = migraphx::argument(ys, y.data());
        pp["cond"]   = migraphx::argument(migraphx::shape{migraphx::shape::bool_type, {1}}, &cond);
        auto results = p.eval(pp);
        EXPECT(results.size() == 2);

        const auto& src = cond ? x : y;
        std::size_t n   = cond ? 3 : 5;
        EXPECT(results[0].get_shape().lens() == std::vector<std::size_t>{4, n});
        std::vector<float> out;
        results[0].visit([&](auto v) { out.assign(v.begin(), v.end()); });
        std::vector<float> gold(4 * n);
        for(std::size_t i = 0; i < 4; ++i)
            for(std::size_t j = 0; j < n; ++j)
                gold[i * n + j] = src[j * 4 + i];
        EXPECT(out == gold);
        std::vector<int64_t> dims;
        results[1].visit([&](auto v) { dims.assign(v.begin(), v.end()); });
        EXPECT(dims == std::vector<int64_t>{4, static_cast<int64_t>(n)});
    }
}
