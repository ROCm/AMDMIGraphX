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

#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/symbolic_tensor_value.hpp>
#include <limits>
#include "test.hpp"

using migraphx::shape;
using migraphx::symbolic_tensor_value;
using migraphx::sym::lit;
using migraphx::sym::var;

struct unsupported_symbolic_op
{
    std::string name() const { return "unsupported_symbolic_op"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return inputs.front(); }
};

struct wrong_count_symbolic_op
{
    std::string name() const { return "wrong_count_symbolic_op"; }
    shape compute_shape(const std::vector<shape>&) const { return {shape::int64_type, {2}}; }
    std::optional<symbolic_tensor_value>
    symbolic_compute(const shape&,
                     const std::vector<shape>&,
                     const std::vector<std::optional<symbolic_tensor_value>>&) const
    {
        return symbolic_tensor_value{lit(1)};
    }
};

struct incomplete_symbolic_op
{
    std::string name() const { return "incomplete_symbolic_op"; }
    shape compute_shape(const std::vector<shape>&) const { return {shape::int64_type, {2}}; }
    std::optional<symbolic_tensor_value>
    symbolic_compute(const shape&,
                     const std::vector<shape>&,
                     const std::vector<std::optional<symbolic_tensor_value>>&) const
    {
        return symbolic_tensor_value{lit(1), {}};
    }
};

static std::size_t symbolic_identity_calls = 0;

struct counting_symbolic_identity
{
    std::string name() const { return "counting_symbolic_identity"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return inputs.front(); }
    std::optional<symbolic_tensor_value>
    symbolic_compute(const shape& output_shape,
                     const std::vector<shape>&,
                     const std::vector<std::optional<symbolic_tensor_value>>& inputs) const
    {
        ++symbolic_identity_calls;
        return migraphx::pass_through_symbolic_value(output_shape, inputs);
    }
};

TEST_CASE(sym_eval_literal)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(
        migraphx::literal{shape{shape::int64_type, {3}}, std::vector<int64_t>{-1, 2, 3}});
    EXPECT(x->sym_eval() == symbolic_tensor_value{lit(-1), lit(2), lit(3)});
}

TEST_CASE(sym_eval_dimensions_of)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    const auto s = var("S", {1, 16});
    auto x       = mm->add_parameter(
        "x",
        shape{shape::float_type, {shape::dynamic_dimension{lit(1)}, shape::dynamic_dimension{s}}});
    auto dims =
        mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 2}}), x);
    EXPECT(dims->sym_eval() == symbolic_tensor_value{lit(1), s});
}

TEST_CASE(sym_eval_recursive)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    const auto s = var("S", {1, 16});
    auto x       = mm->add_parameter(
        "x",
        shape{shape::float_type, {shape::dynamic_dimension{lit(1)}, shape::dynamic_dimension{s}}});
    auto dims =
        mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 2}}), x);
    auto one =
        mm->add_literal(migraphx::literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{1}});
    auto broadcast_one =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), one);
    auto result = mm->add_instruction(migraphx::make_op("add"), dims, broadcast_one);
    EXPECT(result->sym_eval() == symbolic_tensor_value{lit(2), s + lit(1)});
}

TEST_CASE(sym_eval_diamond_memoized)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(
        migraphx::literal{shape{shape::int64_type, {2}}, std::vector<int64_t>{1, 2}});
    symbolic_identity_calls = 0;
    auto shared             = mm->add_instruction(counting_symbolic_identity{}, x);
    auto result             = mm->add_instruction(migraphx::make_op("add"), shared, shared);
    EXPECT(result->sym_eval() == symbolic_tensor_value{lit(2), lit(4)});
    EXPECT(symbolic_identity_calls == 1);
}

TEST_CASE(sym_eval_static_eval_fallback)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(
        migraphx::literal{shape{shape::int64_type, {2}}, std::vector<int64_t>{1, -2}});
    auto result = mm->add_instruction(migraphx::make_op("neg"), x);
    EXPECT(result->sym_eval() == symbolic_tensor_value{lit(-1), lit(2)});
}

TEST_CASE(sym_eval_unsupported)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto x      = mm->add_parameter("x", shape{shape::int64_type, {2}});
    auto result = mm->add_instruction(unsupported_symbolic_op{}, x);
    EXPECT(not result->sym_eval().has_value());
}

TEST_CASE(sym_eval_unsigned_overflow)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(
        migraphx::literal{shape{shape::uint64_type, {1}},
                          std::vector<uint64_t>{std::numeric_limits<uint64_t>::max()}});
    EXPECT(not x->sym_eval().has_value());
}

TEST_CASE(sym_eval_rejects_incomplete_value)
{
    migraphx::program p;
    auto result = p.get_main_module()->add_instruction(incomplete_symbolic_op{},
                                                       std::vector<migraphx::instruction_ref>{});
    EXPECT(not result->sym_eval().has_value());
}

TEST_CASE(sym_eval_rejects_wrong_element_count)
{
    migraphx::program p;
    auto result = p.get_main_module()->add_instruction(wrong_count_symbolic_op{},
                                                       std::vector<migraphx::instruction_ref>{});
    EXPECT(not result->sym_eval().has_value());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
