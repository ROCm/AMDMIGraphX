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
#include <migraphx/sym_argument.hpp>
#include <limits>
#include "test.hpp"

using migraphx::shape;
using migraphx::sym::lit;
using migraphx::sym::var;
using symbolic_tensor_value = std::vector<migraphx::sym::expr>;

std::optional<symbolic_tensor_value> sym_values(migraphx::instruction_ref ins)
{
    auto result = ins->sym_eval();
    if(not result.has_value())
        return std::nullopt;
    return result->get().to_vector();
}

struct unsupported_symbolic_op
{
    std::string name() const { return "unsupported_symbolic_op"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return inputs.front(); }
};

struct fallback_identity_op
{
    std::string name() const { return "fallback_identity_op"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return inputs.front(); }
    migraphx::argument compute(const shape&, std::vector<migraphx::argument> args) const
    {
        return args.front();
    }
};

struct const_fold_before_symbolic_op
{
    std::string name() const { return "const_fold_before_symbolic_op"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return inputs.front(); }
    migraphx::argument compute(const shape&, std::vector<migraphx::argument> args) const
    {
        return args.front();
    }
    migraphx::sym_argument symbolic_compute(const shape& output_shape,
                                            const std::vector<migraphx::sym_argument>&) const
    {
        return {{lit(99)}, output_shape};
    }
};

struct wrong_count_symbolic_op
{
    std::string name() const { return "wrong_count_symbolic_op"; }
    shape compute_shape(const std::vector<shape>&) const { return {shape::int64_type, {2}}; }
    migraphx::sym_argument symbolic_compute(const shape& output_shape,
                                            const std::vector<migraphx::sym_argument>&) const
    {
        return migraphx::sym_argument{{lit(1)}, output_shape};
    }
};

struct incomplete_symbolic_op
{
    std::string name() const { return "incomplete_symbolic_op"; }
    shape compute_shape(const std::vector<shape>&) const { return {shape::int64_type, {2}}; }
    migraphx::sym_argument symbolic_compute(const shape& output_shape,
                                            const std::vector<migraphx::sym_argument>&) const
    {
        return migraphx::sym_argument{{lit(1), {}}, output_shape};
    }
};

static std::size_t symbolic_identity_calls = 0;

struct counting_symbolic_identity
{
    std::string name() const { return "counting_symbolic_identity"; }
    shape compute_shape(const std::vector<shape>& inputs) const { return inputs.front(); }
    migraphx::sym_argument symbolic_compute(const shape& output_shape,
                                            const std::vector<migraphx::sym_argument>& args) const
    {
        ++symbolic_identity_calls;
        return migraphx::pass_through_sym_argument(output_shape, args);
    }
};

TEST_CASE(sym_eval_literal)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(
        migraphx::literal{shape{shape::int64_type, {3}}, std::vector<int64_t>{-1, 2, 3}});
    EXPECT(sym_values(x) == symbolic_tensor_value{lit(-1), lit(2), lit(3)});
}

TEST_CASE(sym_eval_float_literal)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(
        migraphx::literal{shape{shape::float_type, {3}}, std::vector<float>{-1.5, 2.25, 3.0}});
    EXPECT(sym_values(x) == symbolic_tensor_value{lit(-1.5), lit(2.25), lit(3.0)});
}

TEST_CASE(sym_eval_float_arithmetic_fallback_preserves_rounding)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(
        migraphx::literal{shape{shape::float_type, {1}}, std::vector<float>{16777216.0}});
    auto y =
        mm->add_literal(migraphx::literal{shape{shape::float_type, {1}}, std::vector<float>{1.0}});
    auto result = mm->add_instruction(migraphx::make_op("add"), x, y);
    EXPECT(sym_values(result) == symbolic_tensor_value{lit(16777216.0)});
}

TEST_CASE(sym_eval_literal_preserves_layout)
{
    migraphx::program p;
    auto* mm           = p.get_main_module();
    const shape layout = {shape::int64_type, {2, 2}, {1, 2}};
    auto x = mm->add_literal(migraphx::literal{layout, std::vector<int64_t>{1, 2, 3, 4}});
    const auto result = x->sym_eval();
    EXPECT(result.has_value());
    EXPECT(result->get_shape() == layout);
    EXPECT(result->m_data.size() == layout.element_space());
    EXPECT(result->get().to_vector() == symbolic_tensor_value{lit(1), lit(2), lit(3), lit(4)});
}

TEST_CASE(sym_eval_reshape_materializes_layout)
{
    migraphx::program p;
    auto* mm           = p.get_main_module();
    const shape layout = {shape::int64_type, {2, 2}, {1, 2}};
    auto x       = mm->add_literal(migraphx::literal{layout, std::vector<int64_t>{1, 2, 3, 4}});
    auto reshape = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), x);
    const auto result = reshape->sym_eval();
    const shape expected_shape{shape::int64_type, {4}};
    EXPECT(result.has_value());
    EXPECT(result->get_shape() == expected_shape);
    EXPECT(result->get().to_vector() == symbolic_tensor_value{lit(1), lit(2), lit(3), lit(4)});
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
    EXPECT(sym_values(dims) == symbolic_tensor_value{lit(1), s});
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
    EXPECT(sym_values(result) == symbolic_tensor_value{lit(2), s + lit(1)});
}

TEST_CASE(sym_eval_broadcast_view)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one =
        mm->add_literal(migraphx::literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{1}});
    auto broadcast_one =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), one);
    const auto result = broadcast_one->sym_eval();
    const shape expected_shape{shape::int64_type, {2}, {0}};
    EXPECT(result.has_value());
    EXPECT(result->get_shape() == expected_shape);
    EXPECT(result->m_data.size() == 1);
    EXPECT(result->get().to_vector() == symbolic_tensor_value{lit(1), lit(1)});
}

TEST_CASE(sym_eval_broadcast_eval_fallback)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one =
        mm->add_literal(migraphx::literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{1}});
    auto broadcast_one =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), one);
    auto fallback     = mm->add_instruction(fallback_identity_op{}, broadcast_one);
    const auto result = fallback->sym_eval();
    const shape expected_shape{shape::int64_type, {2}, {0}};
    EXPECT(result.has_value());
    EXPECT(result->get_shape() == expected_shape);
    EXPECT(result->m_data.size() == 1);
    EXPECT(result->get().to_vector() == symbolic_tensor_value{lit(1), lit(1)});
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
    EXPECT(sym_values(result) == symbolic_tensor_value{lit(2), lit(4)});
    EXPECT(symbolic_identity_calls == 1);
}

TEST_CASE(sym_eval_static_eval_fallback)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(
        migraphx::literal{shape{shape::int64_type, {2}}, std::vector<int64_t>{1, -2}});
    auto result = mm->add_instruction(migraphx::make_op("neg"), x);
    EXPECT(sym_values(result) == symbolic_tensor_value{lit(-1), lit(2)});
}

TEST_CASE(sym_eval_const_folds_before_symbolic_compute)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x =
        mm->add_literal(migraphx::literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{7}});
    auto result = mm->add_instruction(const_fold_before_symbolic_op{}, x);
    EXPECT(sym_values(result) == symbolic_tensor_value{lit(7)});
}

TEST_CASE(sym_eval_unsupported)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto x      = mm->add_parameter("x", shape{shape::int64_type, {2}});
    auto result = mm->add_instruction(unsupported_symbolic_op{}, x);
    EXPECT(not result->sym_eval().has_value());
}

TEST_CASE(sym_eval_unsigned_clamps)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(
        migraphx::literal{shape{shape::uint64_type, {1}},
                          std::vector<uint64_t>{std::numeric_limits<uint64_t>::max()}});
    EXPECT(sym_values(x) == symbolic_tensor_value{lit(std::numeric_limits<int64_t>::max())});
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
