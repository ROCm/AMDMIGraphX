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

#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program_verify.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/tmp_dir.hpp>
#include <test.hpp>

static migraphx::program make_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto x   = mm->add_parameter("x", s);
    auto one = mm->add_literal(migraphx::literal{s, {1.0f, 1.0f, 1.0f, 1.0f}});
    auto add = mm->add_instruction(migraphx::make_op("add"), x, one);
    mm->add_debug_symbols(add, {"@verify:not-a-number"});
    auto out = mm->add_instruction(migraphx::make_op("relu"), add);
    mm->add_return({out});
    return p;
}

TEST_CASE(verify_program_outputs)
{
    auto result = migraphx::verify::verify_program(
        make_program(), migraphx::make_target("ref"), migraphx::verify::program_mode::outputs);
    EXPECT(result.passed());
}

TEST_CASE(verify_program_output_mismatch)
{
    migraphx::tmp_dir td{"program_verify_output_mismatch"};
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    mm->add_return({x});

    migraphx::program compiled;
    auto* compiled_mm = compiled.get_main_module();
    auto compiled_x   = compiled_mm->add_parameter("x", s);
    auto neg          = compiled_mm->add_instruction(migraphx::make_op("neg"), compiled_x);
    compiled_mm->add_return({neg});
    auto ref = migraphx::make_target("ref");
    compiled.compile(ref);
    auto path = (td.path / "output_mismatch.mxr").string();
    migraphx::save(compiled, path);

    migraphx::parameter_map inputs{
        {"x", migraphx::literal{s, {-2.0f, -1.0f, 1.0f, 2.0f}}.get_argument()}};
    migraphx::verify::program_options options;
    options.compiled_model = path;
    auto result            = migraphx::verify::verify_program(
        p, ref, migraphx::verify::program_mode::outputs, inputs, options);
    EXPECT(not result.passed());
    EXPECT(result.failures().size() == 1);
    EXPECT(result.results.front().rms_error > 0);
}

TEST_CASE(verify_program_compiled_model_unsupported_mode)
{
    migraphx::verify::program_options options;
    options.compiled_model = "unused.mxr";
    EXPECT(test::throws([&] {
        migraphx::verify::verify_program(make_program(),
                                         migraphx::make_target("ref"),
                                         migraphx::verify::program_mode::layerwise,
                                         {},
                                         options);
    }));
}

TEST_CASE(verify_program_instructions)
{
    auto result = migraphx::verify::verify_program(
        make_program(), migraphx::make_target("ref"), migraphx::verify::program_mode::instructions);
    EXPECT(result.passed());
}

TEST_CASE(verify_program_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::parameter_map inputs{
        {"x", migraphx::literal{s, {-2.0f, -1.0f, 1.0f, 2.0f}}.get_argument()}};
    auto result = migraphx::verify::verify_program(make_program(),
                                                   migraphx::make_target("ref"),
                                                   migraphx::verify::program_mode::reduce,
                                                   inputs);
    EXPECT(result.passed());
}

TEST_CASE(verify_program_bisect)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::parameter_map inputs{
        {"x", migraphx::literal{s, {-2.0f, -1.0f, 1.0f, 2.0f}}.get_argument()}};
    auto result = migraphx::verify::verify_program(make_program(),
                                                   migraphx::make_target("ref"),
                                                   migraphx::verify::program_mode::bisect,
                                                   inputs);
    EXPECT(result.passed());
    EXPECT(not result.failure_step.has_value());
    EXPECT(not result.results.empty());
}

TEST_CASE(verify_program_layers)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::parameter_map inputs{
        {"x", migraphx::literal{s, {-2.0f, -1.0f, 1.0f, 2.0f}}.get_argument()}};
    auto result = migraphx::verify::verify_program(make_program(),
                                                   migraphx::make_target("ref"),
                                                   migraphx::verify::program_mode::layerwise,
                                                   inputs);
    EXPECT(result.passed());
    EXPECT(not result.results.empty());
}

TEST_CASE(verify_program_reduce_exception)
{
    migraphx::shape s{migraphx::shape::float_type, {1}};
    migraphx::parameter_map inputs{{"x", migraphx::literal{s, {-2.0f}}.get_argument()}};
    auto result = migraphx::verify::verify_program(make_program(),
                                                   migraphx::make_target("ref"),
                                                   migraphx::verify::program_mode::reduce,
                                                   inputs);
    EXPECT(not result.passed());
    EXPECT(not result.results.empty());
    EXPECT(result.results.front().exception);
}

TEST_CASE(verify_program_empty_output)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {0}};
    auto x   = mm->add_parameter("x", s);
    auto out = mm->add_instruction(migraphx::make_op("relu"), x);
    mm->add_return({out});
    migraphx::parameter_map inputs{
        {"x", migraphx::literal{s, std::vector<float>{}}.get_argument()}};

    auto outputs = migraphx::verify::verify_program(
        p, migraphx::make_target("ref"), migraphx::verify::program_mode::outputs, inputs);
    EXPECT(outputs.passed());

    auto layers = migraphx::verify::verify_program(
        p, migraphx::make_target("ref"), migraphx::verify::program_mode::layerwise, inputs);
    EXPECT(layers.passed());
    EXPECT(not layers.results.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
