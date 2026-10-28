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
#include <migraphx/operation.hpp>
#include <migraphx/sym_argument.hpp>
#include "test.hpp"

using migraphx::shape;
using migraphx::sym::lit;
using migraphx::sym::var;
using symbolic_tensor_value = std::vector<migraphx::sym::expr>;

using symbolic_inputs = std::vector<std::optional<symbolic_tensor_value>>;

auto symbolic_compute_argument(const migraphx::operation& op,
                               const shape& output_shape,
                               const std::vector<shape>& input_shapes,
                               const symbolic_inputs& input_values)
{
    if(input_shapes.size() != input_values.size())
        return migraphx::sym_argument{};
    std::vector<migraphx::sym_argument> args;
    transform(migraphx::range(input_shapes.size()), std::back_inserter(args), [&](auto i) {
        return input_values[i].has_value()
                   ? migraphx::sym_argument{*input_values[i], input_shapes[i]}
                   : migraphx::sym_argument{{}, input_shapes[i]};
    });
    auto result = op.symbolic_compute(output_shape, args);
    if(not result.empty() and (result.get_shape() != output_shape or not result.valid()))
        return migraphx::sym_argument{};
    return result;
}

auto symbolic_compute(const migraphx::operation& op,
                      const shape& output_shape,
                      const std::vector<shape>& input_shapes,
                      const symbolic_inputs& input_values)
{
    auto result = symbolic_compute_argument(op, output_shape, input_shapes, input_values);
    if(result.empty())
        return std::optional<symbolic_tensor_value>{};
    return std::optional<symbolic_tensor_value>{result.get().to_vector()};
}

TEST_CASE(op_sym_dimensions_of)
{
    const auto s = var("S", {1, 16});
    shape input{shape::float_type,
                {shape::dynamic_dimension{lit(1)},
                 shape::dynamic_dimension{s},
                 shape::dynamic_dimension{lit(4)}}};
    auto result = symbolic_compute(migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 3}}),
                                   shape{shape::int64_type, {2}},
                                   {input},
                                   {std::nullopt});
    EXPECT(result == symbolic_tensor_value{s, lit(4)});
}

TEST_CASE(op_sym_identity)
{
    symbolic_tensor_value value{lit(2), lit(3)};
    EXPECT(symbolic_compute(migraphx::make_op("identity"),
                            shape{shape::int64_type, {2}},
                            {shape{shape::int64_type, {2}}},
                            {value}) == value);
}

TEST_CASE(op_sym_identity_float)
{
    symbolic_tensor_value value{lit(1.5), lit(-2.25)};
    EXPECT(symbolic_compute(migraphx::make_op("identity"),
                            shape{shape::float_type, {2}},
                            {shape{shape::float_type, {2}}},
                            {value}) == value);
}

TEST_CASE(op_sym_squeeze)
{
    symbolic_tensor_value value{lit(2), lit(3)};
    EXPECT(symbolic_compute(migraphx::make_op("squeeze", {{"axes", {0}}}),
                            shape{shape::int64_type, {2}},
                            {shape{shape::int64_type, {1, 2}}},
                            {value}) == value);
}

TEST_CASE(op_sym_unsqueeze)
{
    symbolic_tensor_value value{lit(2), lit(3)};
    EXPECT(symbolic_compute(migraphx::make_op("unsqueeze", {{"axes", {0}}}),
                            shape{shape::int64_type, {1, 2}},
                            {shape{shape::int64_type, {2}}},
                            {value}) == value);
}

TEST_CASE(op_sym_reshape)
{
    symbolic_tensor_value value{lit(2), lit(3)};
    EXPECT(symbolic_compute(migraphx::make_op("reshape"),
                            shape{shape::int64_type, {1, 2}},
                            {shape{shape::int64_type, {2}}},
                            {value}) == value);
}

TEST_CASE(op_sym_convert)
{
    symbolic_tensor_value value{lit(2), lit(3)};
    EXPECT(symbolic_compute(migraphx::make_op("convert", {{"target_type", shape::int64_type}}),
                            shape{shape::int64_type, {2}},
                            {shape{shape::int64_type, {2}}},
                            {value}) == value);
    EXPECT(symbolic_compute(migraphx::make_op("convert", {{"target_type", shape::bool_type}}),
                            shape{shape::bool_type, {2}},
                            {shape{shape::int64_type, {2}}},
                            {symbolic_tensor_value{lit(0), lit(1)}}) ==
           symbolic_tensor_value{lit(0), lit(1)});
}

TEST_CASE(op_sym_gather)
{
    const auto s = var("S", {1, 16});
    auto result  = symbolic_compute(
        migraphx::make_op("gather", {{"axis", 0}}),
        shape{shape::int64_type, {2}},
        {shape{shape::int64_type, {3}}, shape{shape::int64_type, {2}}},
        {symbolic_tensor_value{lit(10), s, lit(30)}, symbolic_tensor_value{lit(-2), lit(0)}});
    EXPECT(result == symbolic_tensor_value{s, lit(10)});
}

TEST_CASE(op_sym_gather_rejects_wrong_output_size)
{
    EXPECT(not symbolic_compute(migraphx::make_op("gather", {{"axis", 0}}),
                                shape{shape::int64_type, {1}},
                                {shape{shape::int64_type, {3}}, shape{shape::int64_type, {2}}},
                                {symbolic_tensor_value{lit(10), lit(20), lit(30)},
                                 symbolic_tensor_value{lit(0), lit(1)}})
                   .has_value());
}

TEST_CASE(op_sym_concat)
{
    auto result =
        symbolic_compute(migraphx::make_op("concat", {{"axis", 0}}),
                         shape{shape::int64_type, {3}},
                         {shape{shape::int64_type, {1}}, shape{shape::int64_type, {2}}},
                         {symbolic_tensor_value{lit(1)}, symbolic_tensor_value{lit(2), lit(3)}});
    EXPECT(result == symbolic_tensor_value{lit(1), lit(2), lit(3)});
}

TEST_CASE(op_sym_concat_rejects_wrong_output_size)
{
    EXPECT(
        not symbolic_compute(migraphx::make_op("concat", {{"axis", 0}}),
                             shape{shape::int64_type, {2}},
                             {shape{shape::int64_type, {1}}, shape{shape::int64_type, {2}}},
                             {symbolic_tensor_value{lit(1)}, symbolic_tensor_value{lit(2), lit(3)}})
                .has_value());
}

TEST_CASE(op_sym_multibroadcast_scalar)
{
    const auto output = shape{shape::int64_type, {2}, {0}};
    const auto inputs = std::vector<shape>{shape{shape::int64_type, {1}}};
    const auto values = symbolic_inputs{symbolic_tensor_value{lit(3)}};
    EXPECT(symbolic_compute(
               migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), output, inputs, values) ==
           symbolic_tensor_value{lit(3), lit(3)});
}

TEST_CASE(op_sym_broadcast_scalar)
{
    const auto output = shape{shape::int64_type, {2}, {0}};
    const auto inputs = std::vector<shape>{shape{shape::int64_type, {1}}};
    const auto values = symbolic_inputs{symbolic_tensor_value{lit(3)}};
    EXPECT(symbolic_compute(migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {2}}}),
                            output,
                            inputs,
                            values) == symbolic_tensor_value{lit(3), lit(3)});
}

TEST_CASE(op_sym_slice_with_attributes)
{
    const auto s = var("S", {1, 16});
    auto result  = symbolic_compute(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {3}}}),
        shape{shape::int64_type, {2}},
        {shape{shape::int64_type, {4}}},
        {symbolic_tensor_value{lit(1), s, lit(3), lit(4)}});
    EXPECT(result == symbolic_tensor_value{s, lit(3)});
}

TEST_CASE(op_sym_slice_with_input_bounds)
{
    const auto s = var("S", {1, 16});
    auto result  = symbolic_compute(migraphx::make_op("slice"),
                                    shape{shape::int64_type, {2}},
                                    {shape{shape::int64_type, {4}},
                                     shape{shape::int64_type, {1}},
                                     shape{shape::int64_type, {1}},
                                     shape{shape::int64_type, {1}}},
                                    {symbolic_tensor_value{lit(1), s, lit(3), lit(4)},
                                     symbolic_tensor_value{lit(-3)},
                                     symbolic_tensor_value{lit(-1)},
                                     symbolic_tensor_value{lit(0)}});
    EXPECT(result == symbolic_tensor_value{s, lit(3)});
}

TEST_CASE(op_sym_slice_rejects_symbolic_bounds)
{
    const auto d = var("D", {1, 16});
    const auto s = var("S", {1, 3});
    EXPECT(not symbolic_compute(migraphx::make_op("slice"),
                                shape{shape::int64_type, {2}},
                                {shape{shape::int64_type, {4}},
                                 shape{shape::int64_type, {1}},
                                 shape{shape::int64_type, {1}},
                                 shape{shape::int64_type, {1}}},
                                {symbolic_tensor_value{lit(1), d, lit(3), lit(4)},
                                 symbolic_tensor_value{s},
                                 symbolic_tensor_value{lit(3)},
                                 symbolic_tensor_value{lit(0)}})
                   .has_value());
}

TEST_CASE(op_sym_slice_rejects_wrong_output_size)
{
    const auto s = var("S", {1, 16});
    EXPECT(not symbolic_compute(
                   migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {3}}}),
                   shape{shape::int64_type, {3}},
                   {shape{shape::int64_type, {4}}},
                   {symbolic_tensor_value{lit(1), s, lit(3), lit(4)}})
                   .has_value());
}

TEST_CASE(op_sym_slice_rejects_non_vector_data)
{
    const auto s = var("S", {1, 16});
    EXPECT(not symbolic_compute(
                   migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                   shape{shape::int64_type, {2}},
                   {shape{shape::int64_type, {2, 2}}},
                   {symbolic_tensor_value{lit(1), s, lit(3), lit(4)}})
                   .has_value());
}

TEST_CASE(op_sym_add)
{
    const auto s = var("S", {1, 16});
    const shape output{shape::int64_type, {2}};
    const std::vector<shape> inputs = {shape{shape::int64_type, {2}},
                                       shape{shape::int64_type, {2}, {0}}};
    const symbolic_inputs values    = {symbolic_tensor_value{s, lit(4)},
                                       symbolic_tensor_value{lit(2)}};

    EXPECT(symbolic_compute(migraphx::make_op("add"), output, inputs, values) ==
           symbolic_tensor_value{s + lit(2), lit(6)});
}

TEST_CASE(op_sym_add_broadcast_view)
{
    const shape output{shape::int64_type, {2, 2}};
    const std::vector<shape> inputs = {output, shape{shape::int64_type, {2, 2}, {0, 1}}};
    const symbolic_inputs values    = {symbolic_tensor_value{lit(1), lit(2), lit(3), lit(4)},
                                       symbolic_tensor_value{lit(10), lit(20)}};
    EXPECT(symbolic_compute(migraphx::make_op("add"), output, inputs, values) ==
           symbolic_tensor_value{lit(11), lit(22), lit(13), lit(24)});
}

TEST_CASE(op_sym_sub)
{
    const auto s = var("S", {1, 16});
    const shape output{shape::int64_type, {2}};
    const std::vector<shape> inputs = {shape{shape::int64_type, {2}},
                                       shape{shape::int64_type, {2}, {0}}};
    const symbolic_inputs values    = {symbolic_tensor_value{s, lit(4)},
                                       symbolic_tensor_value{lit(2)}};

    EXPECT(symbolic_compute(migraphx::make_op("sub"), output, inputs, values) ==
           symbolic_tensor_value{s - lit(2), lit(2)});
}

TEST_CASE(op_sym_mul)
{
    const auto s = var("S", {1, 16});
    const shape output{shape::int64_type, {2}};
    const std::vector<shape> inputs = {shape{shape::int64_type, {2}},
                                       shape{shape::int64_type, {2}, {0}}};
    const symbolic_inputs values    = {symbolic_tensor_value{s, lit(4)},
                                       symbolic_tensor_value{lit(2)}};

    EXPECT(symbolic_compute(migraphx::make_op("mul"), output, inputs, values) ==
           symbolic_tensor_value{s * lit(2), lit(8)});
}

TEST_CASE(op_sym_div)
{
    const auto s = var("S", {1, 16});
    const shape output{shape::int64_type, {2}};
    const std::vector<shape> inputs = {shape{shape::int64_type, {2}},
                                       shape{shape::int64_type, {2}, {0}}};
    const symbolic_inputs values    = {symbolic_tensor_value{s, lit(4)},
                                       symbolic_tensor_value{lit(2)}};

    EXPECT(symbolic_compute(migraphx::make_op("div"), output, inputs, values) ==
           symbolic_tensor_value{s / lit(2), lit(2)});
}

TEST_CASE(op_sym_equal)
{
    const auto s = var("S", {1, 16});
    auto result =
        symbolic_compute(migraphx::make_op("equal"),
                         shape{shape::int64_type, {2}},
                         {shape{shape::int64_type, {2}}, shape{shape::int64_type, {2}}},
                         {symbolic_tensor_value{s, lit(2)}, symbolic_tensor_value{s, lit(3)}});
    EXPECT(result == symbolic_tensor_value{lit(1), lit(0)});
}

TEST_CASE(op_sym_where)
{
    const auto s = var("S", {1, 16});
    auto result  = symbolic_compute(migraphx::make_op("where"),
                                    shape{shape::int64_type, {2}},
                                    {shape{shape::bool_type, {2}},
                                     shape{shape::int64_type, {2}},
                                     shape{shape::int64_type, {2}}},
                                    {symbolic_tensor_value{lit(1), lit(0)},
                                     symbolic_tensor_value{s, lit(2)},
                                     symbolic_tensor_value{lit(3), s}});
    EXPECT(result == symbolic_tensor_value{s, s});
}

TEST_CASE(op_sym_where_broadcast_view)
{
    const shape output{shape::int64_type, {2}};
    auto result = symbolic_compute(migraphx::make_op("where"),
                                   output,
                                   {shape{shape::bool_type, {2}, {0}}, output, output},
                                   {symbolic_tensor_value{lit(1)},
                                    symbolic_tensor_value{lit(2), lit(3)},
                                    symbolic_tensor_value{lit(4), lit(5)}});
    EXPECT(result == symbolic_tensor_value{lit(2), lit(3)});
}

TEST_CASE(op_sym_where_scalar_condition)
{
    const shape output{shape::int64_type, {2}};
    auto result = symbolic_compute(migraphx::make_op("where"),
                                   output,
                                   {shape{shape::bool_type, {1}}, output, output},
                                   {symbolic_tensor_value{lit(0)},
                                    symbolic_tensor_value{lit(2), lit(3)},
                                    symbolic_tensor_value{lit(4), lit(5)}});
    EXPECT(result == symbolic_tensor_value{lit(4), lit(5)});
}

TEST_CASE(op_sym_add_rejects_non_scalar_broadcast)
{
    EXPECT(not symbolic_compute(migraphx::make_op("add"),
                                shape{shape::int64_type, {2, 2}},
                                {shape{shape::int64_type, {2, 2}}, shape{shape::int64_type, {2}}},
                                {symbolic_tensor_value{lit(1), lit(2), lit(3), lit(4)},
                                 symbolic_tensor_value{lit(1), lit(2)}})
                   .has_value());
}

TEST_CASE(op_sym_identity_rejects_short_storage)
{
    EXPECT(symbolic_compute_argument(migraphx::make_op("identity"),
                                     shape{shape::int64_type, {2}},
                                     {shape{shape::int64_type, {2}}},
                                     {symbolic_tensor_value{lit(1)}})
               .empty());
}

TEST_CASE(op_sym_compute_unsupported)
{
    EXPECT(symbolic_compute_argument(migraphx::make_op("neg"),
                                     shape{shape::int64_type, {1}},
                                     {shape{shape::int64_type, {1}}},
                                     {symbolic_tensor_value{lit(1)}})
               .empty());
}

TEST_CASE(op_sym_convert_rejects_unsupported_cases)
{
    EXPECT(not symbolic_compute(migraphx::make_op("convert", {{"target_type", shape::int64_type}}),
                                shape{shape::int64_type, {1}},
                                {shape{shape::int32_type, {1}}},
                                {symbolic_tensor_value{lit(1)}})
                   .has_value());
    EXPECT(not symbolic_compute(migraphx::make_op("convert", {{"target_type", shape::bool_type}}),
                                shape{shape::bool_type, {1}},
                                {shape{shape::int64_type, {1}}},
                                {symbolic_tensor_value{lit(2)}})
                   .has_value());
}

TEST_CASE(op_sym_equal_rejects_indeterminate_comparison)
{
    const auto s = var("S", {1, 16});
    const auto t = var("T", {1, 16});
    EXPECT(not symbolic_compute(migraphx::make_op("equal"),
                                shape{shape::int64_type, {1}},
                                {shape{shape::int64_type, {1}}, shape{shape::int64_type, {1}}},
                                {symbolic_tensor_value{s}, symbolic_tensor_value{t}})
                   .has_value());
}

TEST_CASE(op_sym_where_rejects_symbolic_condition)
{
    const auto s = var("S", {1, 16});
    EXPECT(not symbolic_compute(migraphx::make_op("where"),
                                shape{shape::int64_type, {1}},
                                {shape{shape::bool_type, {1}},
                                 shape{shape::int64_type, {1}},
                                 shape{shape::int64_type, {1}}},
                                {symbolic_tensor_value{s},
                                 symbolic_tensor_value{lit(1)},
                                 symbolic_tensor_value{lit(2)}})
                   .has_value());
}

TEST_CASE(op_sym_div_rejects_possible_zero)
{
    const auto z = var("Z", {-1, 1});
    EXPECT(not symbolic_compute(migraphx::make_op("div"),
                                shape{shape::int64_type, {1}},
                                {shape{shape::int64_type, {1}}, shape{shape::int64_type, {1}}},
                                {symbolic_tensor_value{lit(8)}, symbolic_tensor_value{z}})
                   .has_value());
}

TEST_CASE(op_sym_gather_rejects_out_of_bounds_index)
{
    EXPECT(
        not symbolic_compute(migraphx::make_op("gather", {{"axis", 0}}),
                             shape{shape::int64_type, {1}},
                             {shape{shape::int64_type, {2}}, shape{shape::int64_type, {1}}},
                             {symbolic_tensor_value{lit(1), lit(2)}, symbolic_tensor_value{lit(2)}})
                .has_value());
}

TEST_CASE(op_sym_concat_rejects_nonvector_input)
{
    EXPECT(
        not symbolic_compute(migraphx::make_op("concat", {{"axis", 0}}),
                             shape{shape::int64_type, {2}},
                             {shape{shape::int64_type, {1, 1}}, shape{shape::int64_type, {1, 1}}},
                             {symbolic_tensor_value{lit(1)}, symbolic_tensor_value{lit(2)}})
                .has_value());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
