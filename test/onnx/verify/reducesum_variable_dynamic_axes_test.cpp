/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(reducesum_variable_dynamic_axes_test)
{
    using namespace migraphx;
    onnx_options options;
    const std::vector<shape::dynamic_dimension> axes_dims{{0, 3}};
    options.map_dyn_input_dims["axes"] = axes_dims;
    program p = parse_onnx("reducesum_variable_dynamic_axes_verify_test.onnx", options);
    p.compile(make_target("ref"));

    parameter_map pm;
    shape x_shape{shape::float_type, {2, 2, 2}};
    std::vector<float> x(x_shape.elements());
    std::iota(x.begin(), x.end(), 0);
    pm["x"] = argument(x_shape, x.data());

    std::vector<int64_t> axes{1};
    pm["axes"] = argument(shape{shape::int64_type, {1}}, axes.data());

    auto result = p.eval(pm);
}
