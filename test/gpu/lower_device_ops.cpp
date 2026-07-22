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
#include <migraphx/gpu/lower_device_ops.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <test.hpp>
#include <algorithm>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::lower_device_ops{}, migraphx::dead_code_elimination{}});
}

static migraphx::operation precompile(const migraphx::operation& op,
                                      std::size_t additional_args = 0)
{
    return migraphx::make_op(
        "gpu::precompile_op",
        {{"op", migraphx::to_value(op)}, {"additional_args", additional_args}});
}

static migraphx::module make_module(const migraphx::operation& op,
                                    const std::vector<migraphx::shape>& shapes)
{
    migraphx::module m;
    std::vector<migraphx::instruction_ref> args;
    args.reserve(shapes.size());
    for(std::size_t i = 0; i < shapes.size(); ++i)
        args.push_back(m.add_parameter("x" + std::to_string(i), shapes[i]));
    m.add_return({m.add_instruction(op, args)});
    return m;
}

static void check_lowered(const migraphx::operation& op, const std::vector<migraphx::shape>& shapes)
{
    auto m = make_module(op, shapes);
    run_pass(m);
    EXPECT(m == make_module(precompile(op), shapes));
}

TEST_CASE(lower_hip_copy)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    check_lowered(migraphx::make_op("hip::copy"), {s, s});
}

TEST_CASE(lower_hip_fill)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    check_lowered(migraphx::make_op("hip::fill", {{"value", 0}}), {s});
}

TEST_CASE(lower_gpu_contiguous)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto t = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
        auto alloc = m1.add_instruction(
            migraphx::make_op("allocate",
                              {{"shape",
                                migraphx::to_value(migraphx::shape{migraphx::shape::float_type,
                                                                   t->get_shape().lens()})}}));
        m1.add_return({m1.add_instruction(migraphx::make_op("gpu::contiguous"), t, alloc)});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        auto t = m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
        auto alloc = m2.add_instruction(
            migraphx::make_op("allocate",
                              {{"shape",
                                migraphx::to_value(migraphx::shape{migraphx::shape::float_type,
                                                                   t->get_shape().lens()})}}));
        m2.add_return(
            {m2.add_instruction(precompile(migraphx::make_op("contiguous"), 1), t, alloc)});
    }

    EXPECT(m1 == m2);
}

// Dynamic shapes must be left untouched.
TEST_CASE(lower_hip_copy_dynamic_noop)
{
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {3, 3}}};
    auto m        = make_module(migraphx::make_op("hip::copy"), {s, s});
    auto expected = m;
    run_pass(m);
    EXPECT(m == expected);
}

static migraphx::module
make_concat_precompile_module(const std::vector<migraphx::shape>& input_shapes,
                              migraphx::shape output_shape,
                              std::size_t axis)
{
    migraphx::module m;
    std::vector<migraphx::instruction_ref> params;
    params.reserve(input_shapes.size());
    for(const auto& s : input_shapes)
        params.push_back(m.add_parameter("x" + std::to_string(params.size()), s));
    auto alloc = m.add_instruction(
        migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(output_shape)}}));
    params.push_back(alloc);
    m.add_return(
        {m.add_instruction(precompile(migraphx::make_op("concat", {{"axis", axis}}), 1), params)});
    return m;
}

static std::size_t count_instructions(const migraphx::module& m, const std::string& name)
{
    return std::count_if(m.begin(), m.end(), [&](const auto& ins) { return ins.name() == name; });
}

TEST_CASE(lower_dynamic_concat_inserts_copy_to_gpu)
{
    migraphx::shape x{migraphx::shape::float_type, {{1, 4}, {3, 3}}};
    migraphx::shape y{migraphx::shape::float_type, {{2, 4}, {3, 3}}};
    migraphx::shape out{migraphx::shape::float_type, {{3, 8}, {3, 3}}};

    auto m = make_concat_precompile_module({x, y}, out, 0);
    run_pass(m);
    EXPECT(count_instructions(m, "hip::copy_to_gpu") == 2);
}

TEST_CASE(lower_static_concat_skips_copy_to_gpu)
{
    migraphx::shape x{migraphx::shape::float_type, {2, 3}};
    migraphx::shape y{migraphx::shape::float_type, {2, 3}};
    migraphx::shape out{migraphx::shape::float_type, {2, 6}};

    auto m = make_concat_precompile_module({x, y}, out, 1);
    run_pass(m);
    EXPECT(count_instructions(m, "hip::copy_to_gpu") == 0);
}

// End-to-end: the hip::fill kernel compiles and fills the buffer in-place.
TEST_CASE(hip_fill_kernel_runs)
{
    migraphx::shape s{migraphx::shape::float_type, {5, 2}};

    migraphx::gpu::context ctx;
    auto co = migraphx::gpu::compile_op("hip::fill", ctx, {s}, {{"value", 7}});

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto y   = mm->add_parameter("output", s);
    mm->add_instruction(co, y);
    p.compile(migraphx::make_target("gpu"), migraphx::compile_options{});

    auto result =
        migraphx::gpu::from_gpu(p.eval({{"output", migraphx::gpu::allocate_gpu(s)}}).front());

    std::vector<float> expected(s.elements(), 7.0f);
    EXPECT(result == migraphx::literal{s, expected}.get_argument());
}

TEST_CASE(lower_hip_fill_tuple)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape tup{std::vector<migraphx::shape>{s, s}};
    check_lowered(migraphx::make_op("hip::fill", {{"value", 0}}), {tup});
}

TEST_CASE(hip_fill_tuple_kernel_compiles)
{
    migraphx::shape s{migraphx::shape::float_type, {5, 2}};
    migraphx::shape tup{std::vector<migraphx::shape>{s, s}};

    migraphx::gpu::context ctx;
    auto co = migraphx::gpu::compile_op("hip::fill", ctx, {tup}, {{"value", 7}});

    EXPECT(co.name() == "gpu::code_object");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
