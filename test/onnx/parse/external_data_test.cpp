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
#include <migraphx/register_target.hpp>

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

TEST_CASE(create_program_with_weights_test)
{
    migraphx::onnx_options options;
    options.skip_unknown_operators         = true;
    options.external_weights_as_parameters = true;
    auto template_prog = read_onnx("external_data_test.onnx", options);

    const auto& weight_map = template_prog.get_external_weight_map();
    EXPECT(not weight_map.empty());

    // The weight files live in the external_data_path used by read_onnx
    static auto files{::onnx_files()};
    static std::string base_dir = read_weight_files(files);

    auto baked =
        migraphx::create_program_with_weights(template_prog, base_dir, migraphx::make_target("ref"));

    // Baked program should have no external weight map
    EXPECT(baked.get_external_weight_map().empty());

    // Weight parameters should be gone -- only "input" remains
    auto baked_params = baked.get_parameter_shapes();
    for(const auto& entry : weight_map)
    {
        EXPECT(baked_params.count(entry.first) == 0);
    }
    EXPECT(baked_params.count("input") > 0);

    // Baked program should match the standard (literal-based) parse
    auto reference = create_external_data_prog();
    EXPECT(baked == reference);
}
