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
#include <migraphx/cpu/target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/program_verify.hpp>
#include <migraphx/tmp_dir.hpp>
#include <test.hpp>

// The cpu outputs are views into memory owned by the compiled program, so they
// must stay valid until they are compared with the ref outputs.
TEST_CASE(verify_program_outputs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto a   = mm->add_parameter("a", migraphx::shape{migraphx::shape::float_type, {4, 5}});
    auto b   = mm->add_parameter("b", migraphx::shape{migraphx::shape::float_type, {5, 3}});
    mm->add_return({mm->add_instruction(migraphx::make_op("dot"), a, b)});

    auto result = migraphx::verify::verify_program(
        p, migraphx::cpu::target{}, migraphx::verify::program_mode::outputs);
    EXPECT(result.passed());
}

TEST_CASE(eval_external_weights)
{
    migraphx::tmp_dir td{"cpu-external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::shape s{migraphx::shape::float_type, {4, 16}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto w   = mm->add_literal(migraphx::generate_literal(s, 1));
    mm->add_return({mm->add_instruction(migraphx::make_op("add"), x, w)});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p, filename, options);
    auto loaded = migraphx::load(filename);

    migraphx::parameter_map params{{"x", migraphx::generate_argument(s, 2)}};
    p.compile(migraphx::cpu::target{});
    loaded.compile(migraphx::cpu::target{});
    EXPECT(loaded.eval(params).back() == p.eval(params).back());
}

TEST_CASE(eval_reused_external_weights)
{
    migraphx::tmp_dir td{"cpu-external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::shape s{migraphx::shape::float_type, {4, 16}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto w   = mm->add_literal(migraphx::generate_literal(s, 1));
    mm->add_return({mm->add_instruction(migraphx::make_op("add"), x, w)});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p, filename, options);
    auto reused_file = (td.path / "reused.mxr").string();
    migraphx::file_options reuse;
    reuse.reuse_external_data = true;
    migraphx::save(migraphx::load(filename), reused_file, reuse);
    auto loaded = migraphx::load(reused_file);

    migraphx::parameter_map params{{"x", migraphx::generate_argument(s, 2)}};
    p.compile(migraphx::cpu::target{});
    loaded.compile(migraphx::cpu::target{});
    EXPECT(loaded.eval(params).back() == p.eval(params).back());
}

TEST_CASE(eval_inline_weights)
{
    migraphx::tmp_dir td{"cpu-external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::shape s{migraphx::shape::float_type, {4, 16}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto w   = mm->add_literal(migraphx::generate_literal(s, 1));
    mm->add_return({mm->add_instruction(migraphx::make_op("add"), x, w)});

    migraphx::save(p, filename);
    auto loaded = migraphx::load(filename);

    migraphx::parameter_map params{{"x", migraphx::generate_argument(s, 2)}};
    p.compile(migraphx::cpu::target{});
    loaded.compile(migraphx::cpu::target{});
    EXPECT(loaded.eval(params).back() == p.eval(params).back());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
