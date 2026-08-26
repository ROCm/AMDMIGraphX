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

#include <migraphx/make_op.hpp>
#include <migraphx/op/allocate.hpp>
#include <migraphx/op/dyn_slice.hpp>
#include <migraphx/op/dynamic_range.hpp>
#include <migraphx/op/eval_expr_from_shape.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym_substitute.hpp>
#include "test.hpp"

using migraphx::shape;
using migraphx::sym::var;

static migraphx::symbol_map seq_len(std::size_t value)
{
    return {{migraphx::sym::as_symbol(var("seq_len", {1, 64})), value}};
}

TEST_CASE(dyn_slice_starts_and_ends)
{
    auto seq               = var("seq_len", {1, 64});
    migraphx::operation op = migraphx::make_op(
        "dyn_slice",
        {{"axes", std::vector<int64_t>{1}},
         {"starts", migraphx::to_value(std::vector<migraphx::sym::expr>{seq})},
         {"ends", migraphx::to_value(std::vector<migraphx::sym::expr>{seq + 4})}});

    auto specialized = migraphx::any_cast<migraphx::op::dyn_slice>(op.to_static(seq_len(8)));

    EXPECT(specialized.starts.at(0).eval_uint({}) == 8);
    EXPECT(specialized.ends.at(0).eval_uint({}) == 12);
}

TEST_CASE(dynamic_range_output_dim)
{
    auto seq               = var("seq_len", {1, 64});
    migraphx::operation op = migraphx::make_op(
        "dynamic_range", {{"output_dim", migraphx::to_value(shape::dynamic_dimension{seq})}});

    auto specialized = migraphx::any_cast<migraphx::op::dynamic_range>(op.to_static(seq_len(8)));

    EXPECT(specialized.output_dim.has_value());
    EXPECT(not specialized.output_dim->is_symbolic());
    EXPECT(specialized.output_dim->get_interval().min == 8);
    EXPECT(specialized.output_dim->get_interval().max == 8);
}

TEST_CASE(allocate_shape_becomes_static)
{
    auto seq = var("seq_len", {1, 64});
    shape symbolic{shape::float_type,
                   {shape::dynamic_dimension{seq}, shape::dynamic_dimension{4, 4}}};
    EXPECT(symbolic.dynamic());
    migraphx::operation op =
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(symbolic)}});

    auto specialized = migraphx::any_cast<migraphx::op::allocate>(op.to_static(seq_len(8)));

    EXPECT(specialized.s.has_value());
    EXPECT(not specialized.s->dynamic());
    EXPECT(specialized.s->lens() == std::vector<std::size_t>{8, 4});
}

TEST_CASE(eval_expr_from_shape_expressions)
{
    auto seq               = var("seq_len", {1, 64});
    migraphx::operation op = migraphx::make_op(
        "eval_expr_from_shape",
        {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{seq, seq * 2})}});

    auto specialized =
        migraphx::any_cast<migraphx::op::eval_expr_from_shape>(op.to_static(seq_len(8)));

    EXPECT(migraphx::sym::find_variables(specialized.expressions.at(0)).empty());
    EXPECT(specialized.expressions.at(0).eval_uint({}) == 8);
    EXPECT(specialized.expressions.at(1).eval_uint({}) == 16);
}

// The default has to be safe to call on anything, since the specialization pass applies it to
// every instruction it clones.
TEST_CASE(operation_without_symbols_is_unchanged)
{
    migraphx::operation op = migraphx::make_op("add");
    EXPECT(op.to_static(seq_len(8)) == op);

    migraphx::operation slice = migraphx::make_op("slice",
                                                  {{"axes", std::vector<int64_t>{0}},
                                                   {"starts", std::vector<int64_t>{1}},
                                                   {"ends", std::vector<int64_t>{3}}});
    EXPECT(slice.to_static(seq_len(8)) == slice);
}

// A dynamic dimension with no symbol cannot be resolved by a symbol map, so it must survive
// rather than throw or collapse to an arbitrary size.
TEST_CASE(non_symbolic_dynamic_shape_survives)
{
    shape ranged{shape::float_type, {{1, 4}, {4, 4}}};
    migraphx::operation op = migraphx::make_op("allocate", {{"shape", migraphx::to_value(ranged)}});

    auto specialized = migraphx::any_cast<migraphx::op::allocate>(op.to_static(seq_len(8)));

    EXPECT(specialized.s.has_value());
    EXPECT(*specialized.s == ranged);
}

// Substitution is total, so a map that resolves one symbol leaves the others alone instead of
// failing the whole operation.
TEST_CASE(partial_symbol_map_leaves_other_symbols)
{
    auto seq               = var("seq_len", {1, 64});
    auto batch             = var("batch", {1, 8});
    migraphx::operation op = migraphx::make_op(
        "dyn_slice",
        {{"axes", std::vector<int64_t>{0, 1}},
         {"starts", migraphx::to_value(std::vector<migraphx::sym::expr>{batch, seq})},
         {"ends", migraphx::to_value(std::vector<migraphx::sym::expr>{batch + 1, seq + 1})}});

    auto specialized = migraphx::any_cast<migraphx::op::dyn_slice>(op.to_static(seq_len(8)));

    EXPECT(not migraphx::sym::find_variables(specialized.starts.at(0)).empty());
    EXPECT(specialized.starts.at(1).eval_uint({}) == 8);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
