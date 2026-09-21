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
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
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

static migraphx::instruction_ref add_precompile_layout(migraphx::module& m,
                                                       migraphx::instruction_ref input,
                                                       const std::vector<int64_t>& permutation)
{
    auto op    = migraphx::make_op("layout", {{"permutation", permutation}});
    auto alloc = m.add_instruction(migraphx::make_op(
        "allocate", {{"shape", migraphx::to_value(op.compute_shape({input->get_shape()}))}}));
    return m.add_instruction(
        migraphx::make_op("gpu::precompile_op", {{"op", migraphx::to_value(op)}}), input, alloc);
}

TEST_CASE(lower_standard_reshape)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", input_shape);
        auto r = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 4}}}), x);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", input_shape);
        auto r = m2.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {6, 4}}}), x);
        m2.add_return({r});
    }
    EXPECT(m1 == m2);
}

// The 2 input form carries its target shape on the output buffer, which no GPU copy op can
// honor for a rank changing reshape. The matcher does not accept it, so it survives the
// pass untouched rather than being lowered to a copy that reports the input shape.
TEST_CASE(output_buffer_reshape_is_not_lowered)
{
    auto build = [](migraphx::module& m) {
        auto x      = m.add_parameter("x", {migraphx::shape::float_type, {2, 3, 4}});
        auto output = m.add_parameter("output", {migraphx::shape::float_type, {6, 4}});
        auto r      = m.add_instruction(migraphx::make_op("reshape"), x, output);
        m.add_return({r});
    };

    migraphx::module m1;
    build(m1);
    run_pass(m1);

    migraphx::module m2;
    build(m2);
    EXPECT(m1 == m2);
}

TEST_CASE(lower_range_dynamic_reshape)
{
    using dd = migraphx::shape::dynamic_dimension;
    migraphx::shape input_shape{migraphx::shape::float_type, {dd{1, 4}, dd{24, 24}}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", input_shape);
        auto r = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 24}}}), x);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", input_shape);
        auto c = add_contiguous(m2, x);
        auto r = m2.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {0, 24}}}), c);
        m2.add_return({r});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(keep_required_contiguous)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}, {4, 8, 1}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", input_shape);
        auto c = add_contiguous(m1, x);
        auto r = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 4}}}), c);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", input_shape);
        auto c = add_contiguous(m2, x);
        auto r = m2.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {6, 4}}}), c);
        m2.add_return({r});
    }
    EXPECT(m1 == m2);
}

// The input is transposed so case 1 cannot alias it, but the backwards derivation lands on
// the identity permutation. Case 2 declines that and case 3 emits gpu::contiguous, which is
// the same copy without the jit compile a layout would cost.
TEST_CASE(lower_standard_result_with_copy)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {3, 2, 4}, {4, 12, 1}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", input_shape);
        auto r = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 4}}}), x);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", input_shape);
        auto c = add_contiguous(m2, x);
        auto r = m2.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {6, 4}}}), c);
        m2.add_return({r});
    }
    EXPECT(m1 == m2);
    EXPECT(std::prev(m1.end())->inputs().front()->get_shape().standard());
}

TEST_CASE(propagate_reshape_layout)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {1, 1, 1024, 1024}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", input_shape);
        auto r1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 256, 4, 256, 4}}}), x);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 4, 1, 3}}}), r1);
        auto c = add_contiguous(m1, t);
        auto r2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 16, 256, 256}}}), c);
        m1.add_return({add_contiguous(m1, r2)});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", input_shape);
        auto r1 = m2.add_instruction(
            migraphx::make_op("reshape_lazy", {{"dims", {1, 256, 4, 256, 4}}}), x);
        auto t = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 4, 1, 3}}}), r1);
        auto l = add_precompile_layout(m2, t, {0, 3, 4, 1, 2});
        auto r2 =
            m2.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 16, 256, 256}}}), l);
        m2.add_return({add_contiguous(m2, r2)});
    }
    EXPECT(m1 == m2);

    // Pin the layout the view lands on. A module compare cannot catch a stride change that
    // both sides make together, since both derive their strides from the same ops.
    migraphx::shape expected_shape{
        migraphx::shape::float_type, {1, 16, 256, 256}, {1048576, 1, 4096, 16}};
    EXPECT(std::prev(m1.end())->inputs().front()->inputs().front()->get_shape() == expected_shape);
}

// The singleton dims carry arbitrary strides, but the two elements still sit at offsets
// 0 and 1, so dropping a singleton is a pure restriding. eliminate_contiguous drops the
// incoming copy and case 1 aliases the parameter directly; no copy is needed at all.
TEST_CASE(singleton_dims_alias_without_copy)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {1, 1, 2}, {1, 2, 1}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", input_shape);
        auto c = add_contiguous(m1, x);
        auto r = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2}}}), c);
        m1.add_return({add_contiguous(m1, r)});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", input_shape);
        auto r = m2.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 2}}}), x);
        m2.add_return({add_contiguous(m2, r)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(lower_dependent_reshapes)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {1, 4}, {1, 1}};

    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", input_shape);
        auto c  = add_contiguous(m1, x);
        auto r1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4}}}), c);
        auto r2 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2}}}), r1);
        m1.add_return({add_contiguous(m1, r2)});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", input_shape);
        auto r1 = m2.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 4}}}), x);
        auto r2 = m2.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {2, 2}}}), r1);
        m2.add_return({add_contiguous(m2, r2)});
    }
    EXPECT(m1 == m2);
}

// Not converted to a module compare: the expected module would have to name the layout
// permutation, and deriving that by hand from a from_permutation input is exactly the
// thing the test is checking.
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
