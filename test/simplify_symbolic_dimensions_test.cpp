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

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/simplify_symbolic_dimensions.hpp>

#include "test.hpp"

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m, {migraphx::simplify_symbolic_dimensions{}, migraphx::dead_code_elimination{}});
}

// A pointwise operation does not prove that an external parameter dimension and an internal
// data-dependent dimension have the same runtime value.
TEST_CASE(simplify_symbolic_dimensions_preserve_parameter)
{
    using migraphx::sym::var;

    const auto parameter_count = var("parameter_count", {1, 8});
    const auto internal_count  = var("internal_count", {1, 8});
    const migraphx::shape parameter_shape{
        migraphx::shape::int64_type,
        std::vector<migraphx::shape::dynamic_dimension>{{parameter_count}}};

    migraphx::module m1;
    {
        auto parameter = m1.add_parameter("parameter", parameter_shape);
        auto start =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto delta =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto range = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{internal_count})}}),
            start,
            limit,
            delta);
        auto add = m1.add_instruction(migraphx::make_op("add"), parameter, range);
        m1.add_return({add});
    }
    auto expected = m1;
    run_pass(m1);
    EXPECT(m1 == expected);
}

// Equal range inputs do not imply equal output lengths when the operators clamp them to different
// capacities.
TEST_CASE(simplify_symbolic_dimensions_distinct_range_caps)
{
    using migraphx::sym::var;

    const auto short_count = var("short_count", {0, 4});
    const auto long_count  = var("long_count", {0, 8});

    migraphx::module m;
    auto limit = m.add_parameter("limit", {migraphx::shape::int64_type, {1}});
    auto zero  = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
    auto one   = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
    auto short_range = m.add_instruction(
        migraphx::make_op(
            "dynamic_range",
            {{"max_output", 4},
             {"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{short_count})}}),
        zero,
        limit,
        one);
    auto long_range = m.add_instruction(
        migraphx::make_op(
            "dynamic_range",
            {{"max_output", 8},
             {"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{long_count})}}),
        zero,
        limit,
        one);
    auto sum = m.add_instruction(migraphx::make_op("add"), short_range, long_range);
    m.add_return({sum});

    auto expected = m;
    run_pass(m);
    EXPECT(m == expected);
}

// Equivalent dimensions produced inside the graph should also canonicalize symbolic attributes
// that describe the same runtime extent.
TEST_CASE(simplify_symbolic_dimensions_shape_attributes)
{
    using migraphx::sym::var;

    const auto canonical_count = var("canonical_count", {1, 8});
    const auto alternate_count = var("alternate_count", {1, 8});

    migraphx::module m1;
    {
        auto indices = m1.add_parameter("indices", {migraphx::shape::int64_type, {1, 9}});
        auto start =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto delta =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto canonical = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto alternate = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{alternate_count})}}),
            start,
            limit,
            delta);
        // This pointwise edge establishes canonical_count == alternate_count.
        auto add    = m1.add_instruction(migraphx::make_op("add"), canonical, alternate);
        auto sliced = m1.add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {1}},
                 {"starts", {0}},
                 {"ends", migraphx::value::array{migraphx::to_value(alternate_count)}}}),
            indices,
            start,
            limit);
        auto squeezed  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sliced);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_dyn_dims",
                                migraphx::to_value(std::vector<migraphx::shape::dynamic_dimension>{
                                    {alternate_count}})}}),
            delta,
            squeezed);
        m1.add_return({add, broadcast});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto indices = m2.add_parameter("indices", {migraphx::shape::int64_type, {1, 9}});
        auto start =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto delta =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto canonical = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto alternate = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto add    = m2.add_instruction(migraphx::make_op("add"), canonical, alternate);
        auto sliced = m2.add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {1}},
                 {"starts", {0}},
                 {"ends", migraphx::value::array{migraphx::to_value(canonical_count)}}}),
            indices,
            start,
            limit);
        auto squeezed  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sliced);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_dyn_dims",
                                migraphx::to_value(std::vector<migraphx::shape::dynamic_dimension>{
                                    {canonical_count}})}}),
            delta,
            squeezed);
        m2.add_return({add, broadcast});
    }
    EXPECT(m1 == m2);
}

// Rebuild converging paths together so a strict consumer never sees one canonicalized input and
// one stale input while its shape is inferred.
TEST_CASE(simplify_symbolic_dimensions_converging_producers)
{
    using migraphx::sym::var;

    const auto canonical_count = var("canonical_count", {1, 8});
    const auto alternate_count = var("alternate_count", {1, 8});

    migraphx::module m1;
    {
        auto start =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto delta =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto canonical = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto alternate = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{alternate_count})}}),
            start,
            limit,
            delta);
        auto merged = m1.add_instruction(migraphx::make_op("add"), canonical, alternate);
        auto first  = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{alternate_count})}}),
            start,
            limit,
            delta);
        auto second = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{alternate_count})}}),
            start,
            limit,
            delta);
        auto broadcast =
            m1.add_instruction(migraphx::make_op("broadcast", {{"axis", 0}}), first, second);
        m1.add_return({merged, broadcast});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto start =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto delta =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto canonical = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto alternate = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto merged = m2.add_instruction(migraphx::make_op("add"), canonical, alternate);
        auto first  = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto second = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto broadcast =
            m2.add_instruction(migraphx::make_op("broadcast", {{"axis", 0}}), first, second);
        m2.add_return({merged, broadcast});
    }
    EXPECT(m1 == m2);
}

// Equal runtime inputs prove that B and A describe the same range length. The pointwise operation
// between 2*B and independently computed C does not prove that C == 2*A.
TEST_CASE(simplify_symbolic_dimensions_nested_substitutions_idempotent)
{
    using migraphx::sym::var;

    const auto a = var("a", {1, 8});
    const auto b = var("b", {1, 8});
    const auto c = var("c", {2, 16});

    migraphx::module m1;
    {
        auto start =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto concat_limit =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{16}}});
        auto delta =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto a_range = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{a})}}),
            start,
            limit,
            delta);
        auto b_range = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{b})}}),
            start,
            limit,
            delta);
        auto ab = m1.add_instruction(migraphx::make_op("add"), a_range, b_range);
        auto twice_b =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), b_range, b_range);
        auto c_range = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 16},
                 {"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{c})}}),
            start,
            concat_limit,
            delta);
        auto twice_b_c = m1.add_instruction(migraphx::make_op("add"), twice_b, c_range);
        m1.add_return({ab, twice_b_c});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto start =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto concat_limit =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{16}}});
        auto delta =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto a_range = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{a})}}),
            start,
            limit,
            delta);
        auto b_range = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{a})}}),
            start,
            limit,
            delta);
        auto ab = m2.add_instruction(migraphx::make_op("add"), a_range, b_range);
        auto twice_a =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), b_range, b_range);
        auto c_range = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 16},
                 {"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{c})}}),
            start,
            concat_limit,
            delta);
        auto twice_a_c = m2.add_instruction(migraphx::make_op("add"), twice_a, c_range);
        m2.add_return({ab, twice_a_c});
    }
    EXPECT(m1 == m2);

    auto snapshot = m1;
    run_pass(m1);
    EXPECT(m1 == snapshot);
}

// An unrelated symbolic equality must not reinterpret range-based broadcast attributes as
// symbolic expressions.
TEST_CASE(simplify_symbolic_dimensions_preserve_range_broadcast_dimensions)
{
    using migraphx::sym::var;

    const auto canonical_count = var("canonical_count", {1, 8});
    const auto alternate_count = var("alternate_count", {1, 8});
    const std::vector<migraphx::shape::dynamic_dimension> broadcast_dims{{1, 4}, {6, 6}};

    migraphx::module m1;
    {
        auto data   = m1.add_parameter("data", {migraphx::shape::float_type, {1, 6}});
        auto target = m1.add_parameter("target", {migraphx::shape::float_type, broadcast_dims});
        auto start =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto delta =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto canonical = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto alternate = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{alternate_count})}}),
            start,
            limit,
            delta);
        auto merged    = m1.add_instruction(migraphx::make_op("add"), canonical, alternate);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_dyn_dims", migraphx::to_value(broadcast_dims)}}),
            data,
            target);
        m1.add_return({merged, broadcast});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto data   = m2.add_parameter("data", {migraphx::shape::float_type, {1, 6}});
        auto target = m2.add_parameter("target", {migraphx::shape::float_type, broadcast_dims});
        auto start =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto delta =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto canonical = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto alternate = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto merged    = m2.add_instruction(migraphx::make_op("add"), canonical, alternate);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_dyn_dims", migraphx::to_value(broadcast_dims)}}),
            data,
            target);
        m2.add_return({merged, broadcast});
    }
    EXPECT(m1 == m2);
}

// Exercise the remaining mapped operation families with one shared equivalence class. The
// handlers must replace operation attributes while leaving unrelated parameter metadata intact.
TEST_CASE(simplify_symbolic_dimensions_operation_handlers)
{
    using migraphx::sym::var;

    const auto canonical_count = var("canonical_count", {1, 8});
    const auto alternate_count = var("alternate_count", {1, 8});
    const migraphx::shape alternate_shape{
        migraphx::shape::int64_type,
        std::vector<migraphx::shape::dynamic_dimension>{{alternate_count}}};
    const migraphx::shape canonical_shape{
        migraphx::shape::int64_type,
        std::vector<migraphx::shape::dynamic_dimension>{{canonical_count}}};

    migraphx::module m1;
    {
        auto start =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto delta =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto canonical = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto alternate = m1.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{alternate_count})}}),
            start,
            limit,
            delta);
        // The first input chooses canonical_count for every alternate_count attribute below.
        auto add        = m1.add_instruction(migraphx::make_op("add"), canonical, alternate);
        auto allocation = m1.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(alternate_shape)}}));
        auto eval = m1.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions",
                  migraphx::to_value(std::vector<migraphx::sym::expr>{alternate_count})},
                 {"input_shapes",
                  migraphx::to_value(std::vector<migraphx::shape>{alternate_shape})}}),
            alternate);
        auto reshape = m1.add_instruction(
            migraphx::make_op("reshape",
                              {{"dims",
                                migraphx::to_value(std::vector<migraphx::dim_like>{
                                    migraphx::shape::dynamic_dimension{alternate_count}})}}),
            alternate);
        m1.add_return({add, allocation, canonical, alternate, eval, reshape});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto start =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto limit =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
        auto delta =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
        auto canonical = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto alternate = m2.add_instruction(
            migraphx::make_op(
                "dynamic_range",
                {{"max_output", 8},
                 {"output_dim",
                  migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
            start,
            limit,
            delta);
        auto add        = m2.add_instruction(migraphx::make_op("add"), canonical, alternate);
        auto allocation = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(canonical_shape)}}));
        auto eval = m2.add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions",
                  migraphx::to_value(std::vector<migraphx::sym::expr>{canonical_count})},
                 {"input_shapes",
                  migraphx::to_value(std::vector<migraphx::shape>{canonical_shape})}}),
            alternate);
        auto reshape = m2.add_instruction(
            migraphx::make_op("reshape",
                              {{"dims",
                                migraphx::to_value(std::vector<migraphx::dim_like>{
                                    migraphx::shape::dynamic_dimension{canonical_count}})}}),
            alternate);
        m2.add_return({add, allocation, canonical, alternate, eval, reshape});
    }
    EXPECT(m1 == m2);
}

// Runtime extent bindings identify candidates; they do not make independent runtime counts
// equivalent.
TEST_CASE(simplify_symbolic_dimensions_independent_nonzero_counts)
{
    using migraphx::sym::var;

    const auto x_count = var("x_count", {0, 8});
    const auto y_count = var("y_count", {0, 8});

    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::bool_type, {8}});
    auto y    = m.add_parameter("y", {migraphx::shape::bool_type, {8}});
    auto zero = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});

    auto add_nonzero = [&](auto input, const auto& count) {
        auto nonzero = m.add_instruction(migraphx::make_op("nonzero"), input);
        auto indices =
            m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
        auto runtime_count =
            m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
        auto sliced = m.add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {1}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(count)}}}),
            indices,
            zero,
            runtime_count);
        return m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sliced);
    };

    auto x_indices = add_nonzero(x, x_count);
    auto y_indices = add_nonzero(y, y_count);
    auto sum       = m.add_instruction(migraphx::make_op("add"), x_indices, y_indices);
    m.add_return({sum});

    auto expected = m;
    run_pass(m);
    EXPECT(m == expected);
}

// The all-nonzero input rule applies only when the slice is bounded by that NonZero operation's
// runtime count output.
TEST_CASE(simplify_symbolic_dimensions_requires_nonzero_runtime_count)
{
    using migraphx::sym::var;

    const auto input_count  = var("input_count", {0, 8});
    const auto sliced_count = var("sliced_count", {0, 8});
    const auto symbolic_shape =
        migraphx::shape{migraphx::shape::int64_type,
                        std::vector<migraphx::shape::dynamic_dimension>{{input_count}}};

    migraphx::module m;
    auto limit           = m.add_parameter("limit", {migraphx::shape::int64_type, {1}});
    auto unrelated_count = m.add_parameter("unrelated_count", {migraphx::shape::int64_type, {1}});
    auto zero  = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
    auto one   = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
    auto range = m.add_instruction(
        migraphx::make_op(
            "dynamic_range",
            {{"max_output", 8},
             {"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{input_count})}}),
        zero,
        limit,
        one);

    auto storage = m.add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(symbolic_shape)}}));
    auto filled  = m.add_instruction(migraphx::make_op("fill"), one, storage);
    auto nonzero = m.add_instruction(migraphx::make_op("nonzero"), filled);
    auto indices = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
    auto sliced  = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(sliced_count)}}}),
        indices,
        zero,
        unrelated_count);
    auto squeezed = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sliced);
    auto sum      = m.add_instruction(migraphx::make_op("add"), range, squeezed);
    m.add_return({sum});

    auto expected = m;
    run_pass(m);
    EXPECT(m == expected);
}

// Model the MaskRCNN post-processing pattern: NMS produces a runtime proposal count, TopK limits
// those proposals, and NonZero receives a one-filled vector with the TopK extent. The NonZero
// result therefore has that same extent even though its parser expansion introduces a fresh symbol.
TEST_CASE(simplify_symbolic_dimensions_maskrcnn_postprocess)
{
    using migraphx::sym::lit;
    using migraphx::sym::resolve_min;
    using migraphx::sym::var;

    constexpr int64_t n_boxes = 6;
    constexpr int64_t k       = 4;
    const auto nms_count      = var("nms_count", {0, n_boxes});
    const auto nonzero_count  = var("nonzero_count", {0, k});
    const auto nms_extent     = migraphx::sym::min(nms_count, lit(n_boxes));
    const auto topk_count     = resolve_min(lit(k), nms_extent);

    migraphx::module m1;
    {
        auto boxes  = m1.add_parameter("boxes", {migraphx::shape::float_type, {1, n_boxes, 4}});
        auto scores = m1.add_parameter("scores", {migraphx::shape::float_type, {1, 1, n_boxes}});
        auto max_output    = m1.add_parameter("max_output", {migraphx::shape::int64_type, {1}});
        auto iou_threshold = m1.add_parameter("iou_threshold", {migraphx::shape::float_type, {1}});
        auto score_threshold =
            m1.add_parameter("score_threshold", {migraphx::shape::float_type, {1}});

        // Match the parser's NMS expansion: padded tuple outputs are trimmed to num_selected.
        auto nms =
            m1.add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                               boxes,
                               scores,
                               max_output,
                               iou_threshold,
                               score_threshold);
        auto nms_indices =
            m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms);
        auto num_selected =
            m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms);
        auto zero =
            m1.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto nms_slice = m1.add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {0}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(nms_count)}}}),
            nms_indices,
            zero,
            num_selected);
        auto box_column = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {3}}}), nms_slice);
        auto box_indices =
            m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), box_column);
        // Gather NMS-selected scores, then apply TopK so its output extent is min(k, nms_extent).
        auto flat_scores =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {n_boxes}}}), scores);
        auto gathered_scores = m1.add_instruction(
            migraphx::make_op("gather", {{"axis", 0}}), flat_scores, box_indices);
        auto topk = m1.add_instruction(
            migraphx::make_op("topk", {{"k", k}, {"axis", 0}, {"largest", true}}), gathered_scores);
        auto topk_values =
            m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);

        auto one = m1.add_literal(
            migraphx::literal{{migraphx::shape::int64_type, {1}, {0}}, {int64_t{1}}});
        const migraphx::shape mask_shape{
            migraphx::shape::int64_type,
            std::vector<migraphx::shape::dynamic_dimension>{{topk_count}}};
        auto mask_storage = m1.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(mask_shape)}}));
        auto mask = m1.add_instruction(migraphx::make_op("fill"), one, mask_storage);
        mask      = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), mask);

        // Match the parser's NonZero expansion and expose its runtime count as a rank-1 tensor.
        auto nonzero = m1.add_instruction(migraphx::make_op("nonzero"), mask);
        auto nonzero_indices =
            m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
        auto num_nonzero =
            m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
        auto nonzero_slice = m1.add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {1}},
                 {"starts", {0}},
                 {"ends", migraphx::value::array{migraphx::to_value(nonzero_count)}}}),
            nonzero_indices,
            zero,
            num_nonzero);
        auto nonzero_row = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
            nonzero_slice);
        nonzero_row =
            m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), nonzero_row);
        auto nonzero_values = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
            nonzero_row);
        nonzero_values = m1.add_instruction(
            migraphx::make_op("reshape",
                              {{"dims",
                                migraphx::to_value(std::vector<migraphx::dim_like>{
                                    migraphx::shape::dynamic_dimension{nonzero_count}})}}),
            nonzero_values);

        // This is the symbolic collision: both inputs have the same runtime length but carry
        // independently created TopK and NonZero expressions.
        auto merged = m1.add_instruction(migraphx::make_op("add"), topk_values, nonzero_values);
        m1.add_return({merged, nms_slice, topk_values, nonzero_slice});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto boxes  = m2.add_parameter("boxes", {migraphx::shape::float_type, {1, n_boxes, 4}});
        auto scores = m2.add_parameter("scores", {migraphx::shape::float_type, {1, 1, n_boxes}});
        auto max_output    = m2.add_parameter("max_output", {migraphx::shape::int64_type, {1}});
        auto iou_threshold = m2.add_parameter("iou_threshold", {migraphx::shape::float_type, {1}});
        auto score_threshold =
            m2.add_parameter("score_threshold", {migraphx::shape::float_type, {1}});

        auto nms =
            m2.add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                               boxes,
                               scores,
                               max_output,
                               iou_threshold,
                               score_threshold);
        auto nms_indices =
            m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms);
        auto num_selected =
            m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms);
        auto zero =
            m2.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
        auto nms_slice = m2.add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {0}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(nms_count)}}}),
            nms_indices,
            zero,
            num_selected);
        auto box_column = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {3}}}), nms_slice);
        auto box_indices =
            m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), box_column);
        auto flat_scores =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {n_boxes}}}), scores);
        auto gathered_scores = m2.add_instruction(
            migraphx::make_op("gather", {{"axis", 0}}), flat_scores, box_indices);
        auto topk = m2.add_instruction(
            migraphx::make_op("topk", {{"k", k}, {"axis", 0}, {"largest", true}}), gathered_scores);
        auto topk_values =
            m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);

        auto one = m2.add_literal(
            migraphx::literal{{migraphx::shape::int64_type, {1}, {0}}, {int64_t{1}}});
        const migraphx::shape mask_shape{
            migraphx::shape::int64_type,
            std::vector<migraphx::shape::dynamic_dimension>{{topk_count}}};
        auto mask_storage = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(mask_shape)}}));
        auto mask = m2.add_instruction(migraphx::make_op("fill"), one, mask_storage);
        mask      = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), mask);

        auto nonzero = m2.add_instruction(migraphx::make_op("nonzero"), mask);
        auto nonzero_indices =
            m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
        auto num_nonzero =
            m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
        auto nonzero_slice = m2.add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {1}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(topk_count)}}}),
            nonzero_indices,
            zero,
            num_nonzero);
        auto nonzero_row = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
            nonzero_slice);
        nonzero_row =
            m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), nonzero_row);
        auto nonzero_values = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
            nonzero_row);
        nonzero_values = m2.add_instruction(
            migraphx::make_op("reshape",
                              {{"dims",
                                migraphx::to_value(std::vector<migraphx::dim_like>{
                                    migraphx::shape::dynamic_dimension{topk_count}})}}),
            nonzero_values);
        auto merged = m2.add_instruction(migraphx::make_op("add"), topk_values, nonzero_values);
        m2.add_return({merged, nms_slice, topk_values, nonzero_slice});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
