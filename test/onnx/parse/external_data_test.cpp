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
#include <onnx_test_utils.hpp>

TEST_CASE(external_data_test)
{
    migraphx::program p = create_external_data_prog();

    auto prog = optimize_onnx("external_data_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(external_data_as_parameters_test)
{
    migraphx::onnx_options options;
    options.skip_unknown_operators         = true;
    options.external_weights_as_parameters = true;
    auto prog = read_onnx("external_data_test.onnx", options);

    const auto& weight_map = prog.get_external_weight_map();
    EXPECT(not weight_map.empty());

    auto param_shapes = prog.get_parameter_shapes();

    for(const auto& entry : weight_map)
    {
        EXPECT(param_shapes.count(entry.first) > 0);
        EXPECT(entry.second.nbytes > 0);
        EXPECT(not entry.second.filename.empty());
    }

    EXPECT(param_shapes.count("input") > 0);
}
