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

#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(program_verify)
{
    migraphx::shape s{migraphx_shape_float_type, {2, 2}};
    migraphx::program p;
    auto mm = p.get_main_module();
    auto x  = mm.add_parameter("x", s);
    std::vector<float> ones(s.elements(), 1);
    auto one = mm.add_literal(s, ones.data());
    auto add = mm.add_instruction(migraphx::operation{"add"}, {x, one});
    auto out = mm.add_instruction(migraphx::operation{"relu"}, {add});
    mm.add_return({out});

    std::vector<float> input = {-2.0f, -1.0f, 1.0f, 2.0f};
    migraphx::program_parameters params;
    params.add("x", migraphx::argument{s, input.data()});
    migraphx::program_verify_options options;
    migraphx::compile_options compile_options;
    options.set_compile_options(compile_options);
    options.set_rms_tolerance(1e-3);
    options.set_absolute_tolerance(1e-3);
    options.set_relative_tolerance(1e-3);
    options.set_precision(migraphx_program_verify_precision_fp32);
    options.set_ref_use_double(false);
    options.set_name("api");

    auto target = migraphx::target{"ref"};
    auto result = p.verify(target, migraphx_program_verify_mode_outputs, params, options);
    CHECK(result.passed());
    CHECK(result.get_mode() == migraphx_program_verify_mode_outputs);
    CHECK(result.size() == 1);
    CHECK(not result.has_failure_step());
    auto layer = result[0];
    CHECK(layer.get_name() == "api");
    CHECK(layer.get_operator() == "@return");
    CHECK(layer.get_message().empty());
    CHECK(layer.get_index() == 0);
    CHECK(layer.get_rms_error() < 1e-12);
    CHECK(layer.passed());
    CHECK(not layer.threw_exception());

    CHECK(p.verify(target).passed());
    CHECK(p.verify(target, migraphx_program_verify_mode_instructions).passed());
    CHECK(p.verify(target, migraphx_program_verify_mode_reduce, params).passed());
    auto bisect = p.verify(target, migraphx_program_verify_mode_bisect, params);
    CHECK(bisect.passed());
    CHECK(not bisect.has_failure_step());
    CHECK(p.verify(target, migraphx_program_verify_mode_layerwise, params).passed());
}

TEST_CASE(c_api_program_verify_options)
{
    migraphx_program_verify_options_t options  = nullptr;
    migraphx_compile_options_t compile_options = nullptr;
    CHECK(migraphx_program_verify_options_create(&options) == migraphx_status_success);
    CHECK(migraphx_compile_options_create(&compile_options) == migraphx_status_success);
    CHECK(migraphx_program_verify_options_set_compile_options(options, compile_options) ==
          migraphx_status_success);
    CHECK(migraphx_program_verify_options_set_rms_tolerance(options, 1e-3) ==
          migraphx_status_success);
    CHECK(migraphx_program_verify_options_set_absolute_tolerance(options, 1e-3) ==
          migraphx_status_success);
    CHECK(migraphx_program_verify_options_set_relative_tolerance(options, 1e-3) ==
          migraphx_status_success);
    CHECK(migraphx_program_verify_options_set_precision(
              options, migraphx_program_verify_precision_fp32) == migraphx_status_success);
    CHECK(migraphx_program_verify_options_set_ref_use_double(options, false) ==
          migraphx_status_success);
    CHECK(migraphx_program_verify_options_set_compiled_model(options, "") ==
          migraphx_status_success);
    CHECK(migraphx_program_verify_options_set_name(options, "api") == migraphx_status_success);
    CHECK(migraphx_program_verify_options_set_compiled_model(options, nullptr) ==
          migraphx_status_bad_param);
    CHECK(migraphx_program_verify_options_set_name(options, nullptr) == migraphx_status_bad_param);
    CHECK(migraphx_compile_options_destroy(compile_options) == migraphx_status_success);
    CHECK(migraphx_program_verify_options_destroy(options) == migraphx_status_success);
}

TEST_CASE(c_api_program_verify_options_null_handle)
{
    CHECK(migraphx_program_verify_options_set_rms_tolerance(nullptr, 1e-3) ==
          migraphx_status_bad_param);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
