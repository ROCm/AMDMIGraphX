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
#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>

#include <test.hpp>

#include <cstdint>
#include <vector>

using dyn_dim  = migraphx::shape::dynamic_dimension;
using dyn_dims = std::vector<dyn_dim>;

static migraphx::sym::expr lit1() { return migraphx::sym::lit(std::int64_t{1}); }

// compute_broadcasted_dyn_dims

TEST_CASE(broadcast_dyn_dims_range)
{
    // Range-based behaviour is unchanged: 1 broadcasts, equal ranges intersect
    auto out =
        migraphx::compute_broadcasted_dyn_dims(dyn_dims{{1, 1}, {2, 4}}, dyn_dims{{3, 3}, {2, 4}});
    EXPECT(out == dyn_dims{{3, 3}, {2, 4}});
    out = migraphx::compute_broadcasted_dyn_dims(dyn_dims{{2, 4}}, dyn_dims{{3, 3}, {1, 1}});
    EXPECT(out == dyn_dims{{3, 3}, {2, 4}});
    EXPECT(test::throws(
        [] { migraphx::compute_broadcasted_dyn_dims(dyn_dims{{2, 2}}, dyn_dims{{3, 3}}); }));
}

TEST_CASE(broadcast_dyn_dims_symbolic_one)
{
    // A literal 1 broadcasts to a non-fixed variable from either side
    const auto n = migraphx::sym::var("n", {1, 4});
    auto out =
        migraphx::compute_broadcasted_dyn_dims(dyn_dims{dyn_dim{lit1()}}, dyn_dims{dyn_dim{n}});
    EXPECT(out.size() == 1);
    EXPECT(out.front().sym_expr == n);
    out = migraphx::compute_broadcasted_dyn_dims(dyn_dims{dyn_dim{n}}, dyn_dims{dyn_dim{lit1()}});
    EXPECT(out.front().sym_expr == n);
}

TEST_CASE(broadcast_dyn_dims_keeps_variable_fixed_at_one)
{
    // Both dimensions are 1, but only one of them names a variable: keep the variable so
    // later shape computations still see the symbol
    const auto batch = migraphx::sym::var("batch", {1, 1});
    auto out =
        migraphx::compute_broadcasted_dyn_dims(dyn_dims{dyn_dim{lit1()}}, dyn_dims{dyn_dim{batch}});
    EXPECT(out.size() == 1);
    EXPECT(out.front().sym_expr == batch);
    out =
        migraphx::compute_broadcasted_dyn_dims(dyn_dims{dyn_dim{batch}}, dyn_dims{dyn_dim{lit1()}});
    EXPECT(out.front().sym_expr == batch);
    // Two literal 1s stay a literal 1
    out = migraphx::compute_broadcasted_dyn_dims(dyn_dims{dyn_dim{lit1()}},
                                                 dyn_dims{dyn_dim{lit1()}});
    EXPECT(out.front().sym_expr == lit1());
}

TEST_CASE(broadcast_dyn_dims_symbolic_equal)
{
    const auto n = migraphx::sym::var("n", {1, 4});
    auto out     = migraphx::compute_broadcasted_dyn_dims(
        dyn_dims{dyn_dim{n}, dyn_dim{migraphx::sym::lit(std::int64_t{3})}},
        dyn_dims{dyn_dim{n}, dyn_dim{lit1()}});
    EXPECT(out.size() == 2);
    EXPECT(out[0].sym_expr == n);
    EXPECT(out[1].sym_expr == migraphx::sym::lit(std::int64_t{3}));
    // Different variables cannot broadcast against each other
    const auto m = migraphx::sym::var("m", {1, 4});
    EXPECT(test::throws([&] {
        migraphx::compute_broadcasted_dyn_dims(dyn_dims{dyn_dim{n}}, dyn_dims{dyn_dim{m}});
    }));
}

// insert_common_args with symbolic inputs

static migraphx::shape symbolic_shape(const migraphx::sym::expr& n, std::size_t width)
{
    return {migraphx::shape::float_type,
            dyn_dims{dyn_dim{n}, dyn_dim{migraphx::sym::lit(std::int64_t(width))}}};
}

TEST_CASE(common_args_symbolic_shape_donors)
{
    // Every input is broadcast to the common symbolic dims with the other inputs as shape
    // donors, so each broadcast resolves from its inputs once they are static
    const auto n  = migraphx::sym::var("n", {1, 4});
    const auto xs = symbolic_shape(n, 4);
    const migraphx::shape ys{migraphx::shape::float_type, {1, 4}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", xs);
        auto y = m1.add_parameter("y", ys);
        m1.add_return({migraphx::add_common_op(m1, migraphx::make_op("add"), {x, y})});
    }

    migraphx::module m2;
    {
        auto x           = m2.add_parameter("x", xs);
        auto y           = m2.add_parameter("y", ys);
        auto common_dims = migraphx::to_value(xs.dyn_dims());
        auto bx          = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_dyn_dims", common_dims}}), x, y);
        auto by = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_dyn_dims", common_dims}}), y, x);
        m2.add_return({m2.add_instruction(migraphx::make_op("add"), bx, by)});
    }
    EXPECT(m1 == m2);
    EXPECT(m1.get_output_shapes().front() == xs);
}

TEST_CASE(common_args_symbolic_same_input)
{
    // An input used twice donates its own shape so the broadcast keeps the two-input form
    const auto n  = migraphx::sym::var("n", {1, 4});
    const auto xs = symbolic_shape(n, 4);

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", xs);
        m1.add_return({migraphx::add_common_op(m1, migraphx::make_op("mul"), {x, x})});
    }

    migraphx::module m2;
    {
        auto x           = m2.add_parameter("x", xs);
        auto common_dims = migraphx::to_value(xs.dyn_dims());
        auto bx0         = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_dyn_dims", common_dims}}), x, x);
        auto bx1 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_dyn_dims", common_dims}}), x, x);
        m2.add_return({m2.add_instruction(migraphx::make_op("mul"), bx0, bx1)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(common_args_symbolic_common_type)
{
    // The broadcast comes first and the type conversion follows it
    const auto n  = migraphx::sym::var("n", {1, 4});
    const auto xs = symbolic_shape(n, 4);
    const migraphx::shape ys{migraphx::shape::int32_type, {4}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", xs);
        auto y = m1.add_parameter("y", ys);
        m1.add_return({migraphx::add_common_op(m1, migraphx::make_op("add"), {x, y})});
    }

    migraphx::module m2;
    {
        auto x           = m2.add_parameter("x", xs);
        auto y           = m2.add_parameter("y", ys);
        auto common_dims = migraphx::to_value(xs.dyn_dims());
        auto bx          = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_dyn_dims", common_dims}}), x, y);
        auto by = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_dyn_dims", common_dims}}), y, x);
        by = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), by);
        m2.add_return({m2.add_instruction(migraphx::make_op("add"), bx, by)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(common_args_symbolic_no_common_lens)
{
    // Without common_lens only the types are unified
    const auto n  = migraphx::sym::var("n", {1, 4});
    const auto xs = symbolic_shape(n, 4);
    const auto ys = migraphx::shape{migraphx::shape::int32_type, xs.dyn_dims()};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto y    = m1.add_parameter("y", ys);
        auto args = migraphx::insert_common_args(
            m1, m1.end(), {x, y}, migraphx::common_options{.common_lens = false});
        m1.add_return({m1.add_instruction(migraphx::make_op("add"), args)});
    }

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", xs);
        auto y = m2.add_parameter("y", ys);
        y      = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        m2.add_return({m2.add_instruction(migraphx::make_op("add"), x, y)});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
