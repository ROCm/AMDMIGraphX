/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/lower_reshape.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>
#include <algorithm>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m,
                         {migraphx::eliminate_contiguous{"gpu::contiguous"},
                          migraphx::dead_code_elimination{},
                          migraphx::gpu::lower_reshape{},
                          migraphx::dead_code_elimination{}});
}

static migraphx::instruction_ref add_contiguous(migraphx::module& m,
                                                migraphx::instruction_ref input)
{
    const auto& s = input->get_shape();
    auto output_shape =
        s.dynamic() ? migraphx::shape{s.type(), s.dyn_dims()} : migraphx::shape{s.type(), s.lens()};
    auto alloc = m.add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(output_shape)}}));
    return m.add_instruction(migraphx::make_op("gpu::contiguous"), input, alloc);
}

TEST_CASE(lower_standard_reshape)
{
    migraphx::module m;
    auto x              = m.add_parameter("x", {migraphx::shape::float_type, {2, 3, 4}});
    auto r              = m.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 4}}}), x);
    auto expected_shape = r->get_shape();
    m.add_return({r});

    run_pass(m);

    auto result = std::prev(m.end())->inputs().front();
    EXPECT(result->name() == "reshape_lazy");
    EXPECT(result->get_shape() == expected_shape);
}

// The 2 input form only carries its target shape on the output buffer, which no GPU
// copy op can honor for a rank changing reshape. It must be rejected, not lowered to a
// copy that reports the input shape.
TEST_CASE(lower_output_buffer_reshape_throws)
{
    migraphx::module m;
    auto x      = m.add_parameter("x", {migraphx::shape::float_type, {2, 3, 4}});
    auto output = m.add_parameter("output", {migraphx::shape::float_type, {6, 4}});
    auto r      = m.add_instruction(migraphx::make_op("reshape"), x, output);
    m.add_return({r});

    EXPECT(test::throws<migraphx::exception>([&] { run_pass(m); },
                                             "reshape with a runtime output buffer"));
}

TEST_CASE(lower_range_dynamic_reshape)
{
    using dd = migraphx::shape::dynamic_dimension;

    migraphx::module m;
    auto x =
        m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {dd{1, 4}, dd{24, 24}}});
    auto r = m.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 24}}}), x);
    m.add_return({r});

    run_pass(m);

    auto result = std::prev(m.end())->inputs().front();
    EXPECT(result->name() == "reshape_lazy");
    EXPECT(result->inputs().front()->name() == "gpu::contiguous");
}

TEST_CASE(keep_required_contiguous)
{
    migraphx::module m;
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}, {4, 8, 1}};
    auto x = m.add_parameter("x", input_shape);
    auto c = add_contiguous(m, x);
    auto r = m.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 4}}}), c);
    m.add_return({r});

    run_pass(m);

    auto result = std::prev(m.end())->inputs().front();
    EXPECT(result->name() == "reshape_lazy");
    EXPECT(result->inputs().front()->name() == "gpu::contiguous");
}

// The input is transposed so case 1 cannot alias it, but the backwards derivation lands on
// the identity permutation. Case 2 declines that and case 3 emits gpu::contiguous, which is
// the same copy without the jit compile a layout would cost.
TEST_CASE(lower_standard_result_with_copy)
{
    migraphx::module m;
    migraphx::shape input_shape{migraphx::shape::float_type, {3, 2, 4}, {4, 12, 1}};
    auto x = m.add_parameter("x", input_shape);
    auto r = m.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 4}}}), x);
    m.add_return({r});

    run_pass(m);

    auto result = std::prev(m.end())->inputs().front();
    EXPECT(result->name() == "reshape_lazy");
    EXPECT(result->get_shape().standard());
    EXPECT(result->inputs().front()->name() == "gpu::contiguous");
}

TEST_CASE(propagate_reshape_layout)
{
    migraphx::module m;
    auto x  = m.add_parameter("x", {migraphx::shape::float_type, {1, 1, 1024, 1024}});
    auto r1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 256, 4, 256, 4}}}), x);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 4, 1, 3}}}), r1);
    auto c      = add_contiguous(m, t);
    auto r2     = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 16, 256, 256}}}), c);
    auto output = add_contiguous(m, r2);
    m.add_return({output});

    run_pass(m);

    auto output_contiguous = std::prev(m.end())->inputs().front();
    auto reshape           = output_contiguous->inputs().front();
    EXPECT(reshape->name() == "reshape_lazy");
    EXPECT(not reshape->get_shape().standard());
    EXPECT(reshape->inputs().front()->name() == "gpu::precompile_op");
}

// The singleton dims carry arbitrary strides, but the two elements still sit at offsets
// 0 and 1, so dropping a singleton is a pure restriding. eliminate_contiguous drops the
// incoming copy and case 1 aliases the parameter directly; no copy is needed at all.
TEST_CASE(singleton_dims_alias_without_copy)
{
    migraphx::module m;
    migraphx::shape input_shape{migraphx::shape::float_type, {1, 1, 2}, {1, 2, 1}};
    auto x      = m.add_parameter("x", input_shape);
    auto c      = add_contiguous(m, x);
    auto r      = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2}}}), c);
    auto output = add_contiguous(m, r);
    m.add_return({output});

    run_pass(m);

    auto output_contiguous = std::prev(m.end())->inputs().front();
    auto reshape           = output_contiguous->inputs().front();
    EXPECT(reshape->name() == "reshape_lazy");
    EXPECT(reshape->inputs().front() == x);
}

TEST_CASE(lower_dependent_reshapes)
{
    migraphx::module m;
    migraphx::shape input_shape{migraphx::shape::float_type, {1, 4}, {1, 1}};
    auto x      = m.add_parameter("x", input_shape);
    auto c      = add_contiguous(m, x);
    auto r1     = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4}}}), c);
    auto r2     = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2}}}), r1);
    auto output = add_contiguous(m, r2);
    m.add_return({output});

    run_pass(m);

    EXPECT(
        std::none_of(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "reshape"; }));
    EXPECT(std::count_if(m.begin(), m.end(), [](const auto& ins) {
               return ins.name() == "reshape_lazy";
           }) == 2);
}

TEST_CASE(propagate_symbolic_reshape_layout)
{
    using dd = migraphx::shape::dynamic_dimension;
    using migraphx::sym::lit;

    auto n                      = migraphx::sym::var("N", {1, 8});
    migraphx::shape input_shape = migraphx::shape::from_permutation(
        migraphx::shape::float_type,
        {dd{n}, dd{lit(4)}, dd{lit(4)}, dd{lit(256)}, dd{lit(256)}},
        {0, 3, 1, 4, 2});

    migraphx::module m;
    auto x = m.add_parameter("x", input_shape);
    auto r = m.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 16, 256, 256}}}), x);
    m.add_return({r});

    run_pass(m);

    auto reshape = std::prev(m.end())->inputs().front();
    EXPECT(reshape->name() == "reshape_lazy");
    EXPECT(not reshape->get_shape().standard());
    EXPECT(reshape->get_shape().sym_dims() ==
           std::vector<migraphx::sym::expr>{n, lit(16), lit(256), lit(256)});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
