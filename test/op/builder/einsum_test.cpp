/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <op_builder_test_utils.hpp>

static bool test_invalid_input(const std::string& equation             = "",
                               const std::string& expected_msg         = "",
                               const std::vector<std::size_t>& x1_dims = {3, 3},
                               const std::vector<std::size_t>& x2_dims = {3, 3})
{
    migraphx::module mm;

    if(not x1_dims.empty())
        mm.add_parameter("x1", {migraphx::shape::float_type, x1_dims});

    if(not x2_dims.empty())
        mm.add_parameter("x2", {migraphx::shape::float_type, x2_dims});

    migraphx::value options{};
    if(not equation.empty())
    {
        options.insert({"equation", equation});
    }

    return test::throws<migraphx::exception>(
        [&]() { make_op_module("einsum", options, mm.get_parameters()); }, expected_msg);
}

TEST_CASE(einsum_multiple_arrows_negative_op_builder_test)
{
    EXPECT(test_invalid_input("ii,jj->->ij",
                              "einsum op_builder: Einsum equation has multiple '->' symbols"));
}

TEST_CASE(einsum_empty_term_before_arrow_negative_op_builder_test)
{
    EXPECT(
        test_invalid_input("ii,->ij", "einsum op_builder: No term specified before '->' symbol"));
}

TEST_CASE(einsum_multiple_ellipses_negative_op_builder_test)
{
    EXPECT(test_invalid_input(
        "......ii,...jj->...ij",
        "einsum op_builder: Ellipsis can only appear once per einsum equation term"));
}

TEST_CASE(einsum_comma_in_output_negative_op_builder_test)
{
    EXPECT(test_invalid_input(
        "ii,jj->i,j", "einsum op_builder: Einsum equation can't have a ',' symbol in the output"));
}

TEST_CASE(einsum_empty_term_before_comma_negative_op_builder_test)
{
    EXPECT(
        test_invalid_input("ii,,jj->ij", "einsum op_builder: No term specified before ',' symbol"));
}

TEST_CASE(einsum_last_input_missing_negative_op_builder_test)
{
    EXPECT(test_invalid_input("ii,jj,", "einsum op_builder: Last input term is missing"));
}

TEST_CASE(einsum_term_input_mismatch_negative_op_builder_test)
{
    EXPECT(test_invalid_input(
        "ii,jj,kk->ijk",
        "Number of terms in the input equation - 3 does not match the number of inputs 2"));
}

TEST_CASE(einsum_ellipsis_mismatch_negative_op_builder_test)
{
    EXPECT(test_invalid_input("...ii,...jj->...ij",
                              "einsum op_builder: Every occurrence of ellipsis in the equation "
                              "must represent the same number of dimensions",
                              {3, 3, 3},
                              {3, 3, 3, 3}));
}

TEST_CASE(einsum_rank_mismatch_negative_op_builder_test)
{
    EXPECT(test_invalid_input("iik,jj->ij",
                              "einsum op_builder: Number of labels in 1. input_term (iik) does "
                              "not match the rank (2) of corresponding input"));
}

TEST_CASE(einsum_output_surplus_label_negative_op_builder_test)
{
    EXPECT(test_invalid_input("ii,jj->ijk",
                              "einsum op_builder: Output term contains label 107, which is not "
                              "present in any of the input terms"));
}

TEST_CASE(einsum_output_missing_ellipsis_negative_op_builder_test)
{
    EXPECT(test_invalid_input("...ii,...jj->ij",
                              "einsum op_builder: Output term does not contain ellipsis (...) "
                              "even though an input term does",
                              {3, 3, 3},
                              {3, 3, 3}));
}

TEST_CASE(einsum_multiple_diagonals_negative_op_builder_test)
{
    EXPECT(test_invalid_input("iijj->ij",
                              "einsum op_builder: Parsing of equations with more than one "
                              "duplicated labels per input term is not implemented",
                              {3, 3, 3, 3},
                              {}));
}

TEST_CASE(einsum_diagonal_dim_mismatch_negative_op_builder_test)
{
    EXPECT(
        test_invalid_input("ii->i",
                           "einsum op_builder: All duplicate labels have to be the same dimension",
                           {3, 4},
                           {}));
}

TEST_CASE(einsum_right_batch_diagonal_negative_op_builder_test)
{
    EXPECT(test_invalid_input("ii...->i...",
                              "einsum op_builder: Parsing of equations with duplicated labels and "
                              "batch axes that are not the outer-most axes, is not implemented",
                              {3, 3, 3},
                              {}));
}

TEST_CASE(einsum_trace_of_a_matrix_op_builder_test)
{
    const std::vector<size_t> input_shape  = {3, 3};
    const std::vector<size_t> indices_lens = {3, 2};
    const std::vector<size_t> indices      = {0, 0, 1, 1, 2, 2};
    const size_t batch_dims                = 0;
    const std::vector<int64_t> perm        = {0};
    const std::vector<int> unsq_axes       = {};
    const std::vector<int> red             = {0};
    const std::vector<int> sq_axes         = {0};

    migraphx::module mm;

    auto indices_arg = mm.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, indices_lens}, indices});
    auto a  = mm.add_parameter("a", {migraphx::shape::float_type, input_shape});
    auto op = mm.add_instruction(
        migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), a, indices_arg);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), op);
    op = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsq_axes}}), op);
    op = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", red}}), op);
    mm.add_instruction(migraphx::make_op("squeeze", {{"axes", sq_axes}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ii"}}, mm.get_parameters()));
}

TEST_CASE(einsum_extract_diagonal_op_builder_test)
{
    const std::vector<size_t> input_shape  = {3, 3};
    const std::vector<size_t> indices_lens = {3, 2};
    const std::vector<size_t> indices      = {0, 0, 1, 1, 2, 2};
    const size_t batch_dims                = 0;
    const std::vector<int64_t> perm        = {0};
    const std::vector<int> unsq_axes       = {};

    migraphx::module mm;

    auto indices_arg = mm.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, indices_lens}, indices});
    auto a  = mm.add_parameter("a", {migraphx::shape::float_type, input_shape});
    auto op = mm.add_instruction(
        migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), a, indices_arg);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), op);
    op = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsq_axes}}), op);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ii->i"}}, mm.get_parameters()));
}

TEST_CASE(einsum_matmul_op_builder_test)
{
    const std::vector<size_t> input_shape_a  = {3, 5};
    const std::vector<size_t> input_shape_b  = {5, 2};
    const std::vector<int64_t> perm_1        = {0, 1};
    const std::vector<int> unsq_axes_1       = {2};
    const std::vector<int> unsq_axes_2       = {0};
    const std::vector<int64_t> perm_2        = {0, 2, 1};
    const std::vector<int64_t> reshape_dims1 = {1, -1, 5};
    const std::vector<int64_t> reshape_dims2 = {3, 2, 1};
    const std::vector<int> sq_axes           = {1};

    migraphx::module mm;

    auto a = mm.add_parameter("a", {migraphx::shape::float_type, input_shape_a});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, input_shape_b});

    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsq_axes_1}}), tr1);

    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsq_axes_2}}), tr2);

    auto tr3 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_2}}), unsq1);
    auto tr4 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_2}}), unsq2);

    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims1}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims1}}), tr4);
    auto tr5 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_2}}), resh2);
    auto dot = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims2}}), dot);

    auto tr6 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_2}}), resh3);
    auto sq  = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", sq_axes}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij,jk"}}, mm.get_parameters()));
}

TEST_CASE(einsum_matrix_transpose_op_builder_test)
{
    const std::vector<size_t> input_shape_a = {3, 5};
    const std::vector<int64_t> perm_1       = {1, 0};
    const std::vector<int64_t> perm_2       = {0, 1};
    const std::vector<int> unsq_axes        = {};

    migraphx::module mm;

    auto a  = mm.add_parameter("a", {migraphx::shape::float_type, input_shape_a});
    auto op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), a);
    op      = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsq_axes}}), op);
    op      = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_2}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ji"}}, mm.get_parameters()));
}

TEST_CASE(einsum_element_wise_mul_op_builder_test)
{
    const std::vector<size_t> input_shape    = {5};
    const std::vector<int64_t> perm_1        = {0};
    const std::vector<int> unsq_axes_1       = {};
    const std::vector<int> unsq_axes_2       = {};
    const std::vector<int64_t> perm_2        = {0, 2, 1};
    const std::vector<int64_t> reshape_dims1 = {5, -1, 1};
    const std::vector<int64_t> reshape_dims2 = {5};
    const std::vector<int> sq_axes           = {1};

    migraphx::module mm;

    auto a = mm.add_parameter("a", {migraphx::shape::float_type, input_shape});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, input_shape});

    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsq_axes_1}}), tr1);

    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsq_axes_2}}), tr2);

    auto tr3 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), unsq1);
    auto tr4 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), unsq2);

    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims1}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims1}}), tr4);
    auto tr5 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_2}}), resh2);
    auto dot = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims2}}), dot);

    auto tr6 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), resh3);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", perm_1}}), tr6);

    EXPECT(mm == make_op_module("einsum", {{"equation", "i,i->i"}}, mm.get_parameters()));
}
