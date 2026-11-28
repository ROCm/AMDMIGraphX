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

TEST_CASE(dim_param_fixed_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4}});
    mm->add_return({input});

    migraphx::onnx_options opt;
    opt.dim_params = {{"dim0", migraphx::shape::dynamic_dimension{2, 2}},
                      {"dim1", migraphx::shape::dynamic_dimension{4, 4}}};
    auto prog      = read_onnx("dim_param_test.onnx", opt);
    EXPECT(p == prog);
}

TEST_CASE(dim_param_dynamic_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0",
                                   migraphx::shape{migraphx::shape::float_type,
                                                   {migraphx::shape::dynamic_dimension{1, 2},
                                                    migraphx::shape::dynamic_dimension{2, 4}}});
    mm->add_return({input});

    migraphx::onnx_options opt;
    opt.dim_params = {{"dim0", migraphx::shape::dynamic_dimension{1, 2}},
                      {"dim1", migraphx::shape::dynamic_dimension{2, 4}}};
    auto prog      = read_onnx("dim_param_test.onnx", opt);
    EXPECT(p == prog);
}

TEST_CASE(dim_param_symbolic_test)
{
    using migraphx::sym::var;
    // use_symbolic_shapes maps each named dim to a sym::var; dim_params supplies its bounds.
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("0",
                          migraphx::shape{migraphx::shape::float_type,
                                          sym_dims({var("dim0", {1, 2}), var("dim1", {2, 4})})});
    mm->add_return({input});

    migraphx::onnx_options opt;
    opt.use_symbolic_shapes = true;
    opt.dim_params          = {{"dim0", migraphx::shape::dynamic_dimension{1, 2}},
                               {"dim1", migraphx::shape::dynamic_dimension{2, 4}}};
    auto prog               = read_onnx("dim_param_test.onnx", opt);
    EXPECT(p == prog);
}

TEST_CASE(dim_param_symbolic_map_dyn_input_test)
{
    using migraphx::sym::var;
    // With use_symbolic_shapes, a plain-range map_dyn_input_dims override takes its bounds from
    // the override but its symbol name from the model's dim_param for each axis.
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("0",
                          migraphx::shape{migraphx::shape::float_type,
                                          sym_dims({var("dim0", {3, 6}), var("dim1", {5, 10})})});
    mm->add_return({input});

    migraphx::onnx_options opt;
    opt.use_symbolic_shapes     = true;
    opt.map_dyn_input_dims["0"] = {{3, 6}, {5, 10}};
    auto prog                   = read_onnx("dim_param_test.onnx", opt);
    EXPECT(p == prog);
}

TEST_CASE(dim_param_symbolic_map_dyn_input_optimals_test)
{
    using migraphx::sym::var;
    // Optimals on the override are carried onto the synthesized symbol.
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter(
        "0",
        migraphx::shape{migraphx::shape::float_type,
                        sym_dims({var("dim0", {1, 8}, {2, 4}), var("dim1", {5, 10})})});
    mm->add_return({input});

    migraphx::onnx_options opt;
    opt.use_symbolic_shapes     = true;
    opt.map_dyn_input_dims["0"] = {{1, 8, {2, 4}}, {5, 10}};
    auto prog                   = read_onnx("dim_param_test.onnx", opt);
    EXPECT(p == prog);
}

// A dim_param is an arbitrary string but a symbol name has to be an identifier, so the parser
// rewrites it. "batch.size" and "batch_size" sanitize alike, so the second to be seen gains a
// counter rather than silently sharing the first one's symbol.
TEST_CASE(dim_param_symbolic_odd_names_test)
{
    using migraphx::sym::var;
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0",
                                   migraphx::shape{migraphx::shape::float_type,
                                                   sym_dims({var("batch_size", {1, 4}),
                                                             var("batch_size_2", {1, 4}),
                                                             var("_2d", {1, 4})})});
    mm->add_return({input});

    migraphx::onnx_options opt;
    opt.use_symbolic_shapes   = true;
    opt.default_dyn_dim_value = {1, 4};
    auto prog                 = read_onnx("dim_param_odd_names_test.onnx", opt);
    EXPECT(p == prog);
}

TEST_CASE(dim_param_symbolic_map_dyn_input_explicit_sym_test)
{
    using migraphx::sym::var;
    // An override that is already symbolic is honored verbatim, keeping its own symbol names
    // rather than being renamed to the model's dim_params.
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0",
                                   migraphx::shape{migraphx::shape::float_type,
                                                   sym_dims({var("n", {1, 4}), var("m", {2, 8})})});
    mm->add_return({input});

    migraphx::onnx_options opt;
    opt.use_symbolic_shapes     = true;
    opt.map_dyn_input_dims["0"] = sym_dims({var("n", {1, 4}), var("m", {2, 8})});
    auto prog                   = read_onnx("dim_param_test.onnx", opt);
    EXPECT(p == prog);
}
