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
#include <numeric>
#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <migraphx/manage_ptr.hpp>
#include "test.hpp"

TEST_CASE(dynamic_batch_load_and_run_offload)
{
    migraphx::onnx_options o_options;
    migraphx::dynamic_dimensions dyn_dims = {migraphx::dynamic_dimension{1, 4, {2, 4}},
                                             migraphx::dynamic_dimension{3, 3},
                                             migraphx::dynamic_dimension{4, 4},
                                             migraphx::dynamic_dimension{4, 4}};
    o_options.set_dyn_input_parameter_shape("0", dyn_dims);
    dyn_dims = {migraphx::dynamic_dimension{2, 2},
                migraphx::dynamic_dimension{3, 3},
                migraphx::dynamic_dimension{3, 3},
                migraphx::dynamic_dimension{3, 3}};
    o_options.set_dyn_input_parameter_shape("1", dyn_dims);
    auto p             = migraphx::parse_onnx("conv_dynamic_batch_test.onnx", o_options);
    auto shapes_before = p.get_output_shapes();
    migraphx::compile_options c_options;
    c_options.set_offload_copy();
    p.compile(migraphx::target("gpu"), c_options);
    auto out_shapes = p.get_output_shapes();
    EXPECT(out_shapes.size() == 1);
    EXPECT(out_shapes[0].dynamic());

    // batch size = 2
    std::vector<float> a(2 * 3 * 4 * 4, 0.12);
    std::vector<float> c(2 * 3 * 3 * 3, 0.75);
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    pp.add("0",
           migraphx::argument(migraphx::shape(migraphx_shape_float_type, {2, 3, 4, 4}), a.data()));
    pp.add("1",
           migraphx::argument(migraphx::shape(migraphx_shape_float_type, {2, 3, 3, 3}), c.data()));
    auto outputs = p.eval(pp);

    EXPECT(shapes_before.size() == outputs.size());
    EXPECT(bool{outputs.front().get_shape() ==
                migraphx::shape(migraphx_shape_float_type, {2, 2, 2, 2})});
}

TEST_CASE(dynamic_batch_load_and_run_offload_repeat2)
{
    migraphx::onnx_options o_options;
    migraphx::dynamic_dimensions dyn_dims = {migraphx::dynamic_dimension{1, 4, {2, 4}},
                                             migraphx::dynamic_dimension{3, 3},
                                             migraphx::dynamic_dimension{4, 4},
                                             migraphx::dynamic_dimension{4, 4}};
    o_options.set_dyn_input_parameter_shape("0", dyn_dims);
    dyn_dims = {migraphx::dynamic_dimension{2, 2},
                migraphx::dynamic_dimension{3, 3},
                migraphx::dynamic_dimension{3, 3},
                migraphx::dynamic_dimension{3, 3}};
    o_options.set_dyn_input_parameter_shape("1", dyn_dims);
    auto p             = migraphx::parse_onnx("conv_dynamic_batch_test.onnx", o_options);
    auto shapes_before = p.get_output_shapes();
    migraphx::compile_options c_options;
    c_options.set_offload_copy();
    p.compile(migraphx::target("gpu"), c_options);
    auto out_shapes = p.get_output_shapes();
    EXPECT(out_shapes.size() == 1);
    EXPECT(out_shapes[0].dynamic());

    // batch size = 2
    std::vector<float> a(2 * 3 * 4 * 4, 0.12);
    std::vector<float> c(2 * 3 * 3 * 3, 0.75);
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    pp.add("0",
           migraphx::argument(migraphx::shape(migraphx_shape_float_type, {2, 3, 4, 4}), a.data()));
    pp.add("1",
           migraphx::argument(migraphx::shape(migraphx_shape_float_type, {2, 3, 3, 3}), c.data()));
    auto outputs = p.eval(pp);

    EXPECT(shapes_before.size() == outputs.size());
    EXPECT(bool{outputs.front().get_shape() ==
                migraphx::shape(migraphx_shape_float_type, {2, 2, 2, 2})});
}

TEST_CASE(dynamic_batch_load_and_run_offload_repeat3)
{
    migraphx::onnx_options o_options;
    migraphx::dynamic_dimensions dyn_dims = {migraphx::dynamic_dimension{1, 4, {2, 4}},
                                             migraphx::dynamic_dimension{3, 3},
                                             migraphx::dynamic_dimension{4, 4},
                                             migraphx::dynamic_dimension{4, 4}};
    o_options.set_dyn_input_parameter_shape("0", dyn_dims);
    dyn_dims = {migraphx::dynamic_dimension{2, 2},
                migraphx::dynamic_dimension{3, 3},
                migraphx::dynamic_dimension{3, 3},
                migraphx::dynamic_dimension{3, 3}};
    o_options.set_dyn_input_parameter_shape("1", dyn_dims);
    auto p             = migraphx::parse_onnx("conv_dynamic_batch_test.onnx", o_options);
    auto shapes_before = p.get_output_shapes();
    migraphx::compile_options c_options;
    c_options.set_offload_copy();
    p.compile(migraphx::target("gpu"), c_options);
    auto out_shapes = p.get_output_shapes();
    EXPECT(out_shapes.size() == 1);
    EXPECT(out_shapes[0].dynamic());

    // batch size = 2
    std::vector<float> a(2 * 3 * 4 * 4, 0.12);
    std::vector<float> c(2 * 3 * 3 * 3, 0.75);
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    pp.add("0",
           migraphx::argument(migraphx::shape(migraphx_shape_float_type, {2, 3, 4, 4}), a.data()));
    pp.add("1",
           migraphx::argument(migraphx::shape(migraphx_shape_float_type, {2, 3, 3, 3}), c.data()));
    auto outputs = p.eval(pp);

    EXPECT(shapes_before.size() == outputs.size());
    EXPECT(bool{outputs.front().get_shape() ==
                migraphx::shape(migraphx_shape_float_type, {2, 2, 2, 2})});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
