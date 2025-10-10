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

TEST_CASE(einsum_permute_op_builder_test)
{
    migraphx::module mm;
    auto a  = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    op      = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op      = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij->ji"}}, mm.get_parameters()));
}

TEST_CASE(einsum_summation_op_builder_test)
{
    migraphx::module mm;
    auto a  = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    op      = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op      = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), op);
    mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij->"}}, mm.get_parameters()));
}

TEST_CASE(einsum_column_sum_op_builder_test)
{
    migraphx::module mm;
    auto a  = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    op      = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op      = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), op);
    op      = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), op);
    op      = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij->j"}}, mm.get_parameters()));
}

TEST_CASE(einsum_row_sum_op_builder_test)
{
    migraphx::module mm;
    auto a  = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    op      = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op      = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), op);
    op      = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), op);
    op      = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij->i"}}, mm.get_parameters()));
}

TEST_CASE(einsum_matrix_vector_multiplication_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto v     = mm.add_parameter("v", {migraphx::shape::float_type, {3}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), v);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), tr2);
    auto tr3 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq1);
    auto tr4 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1}}}), dot);
    auto tr6 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), resh3);
    auto sq  = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij,j->i"}}, mm.get_parameters()));
}

TEST_CASE(einsum_matrix_matrix_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {2, 3}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 1}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij,kj->ik"}}, mm.get_parameters()));
}

TEST_CASE(einsum_vector_dot_product_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {3}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {3}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr2);
    auto tr3   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), unsq1);
    auto tr4   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1}}}), dot);
    auto tr6   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), resh3);
    mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), tr6);

    EXPECT(mm == make_op_module("einsum", {{"equation", "i,i->"}}, mm.get_parameters()));
}

TEST_CASE(einsum_matrix_dot_product_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {2, 3}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr2);
    auto tr3 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq1);
    auto tr4 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 6}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 6}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1}}}), dot);
    auto tr6 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), resh3);
    mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), tr6);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij,ij->"}}, mm.get_parameters()));
}

TEST_CASE(einsum_hadamard_product_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {2, 3}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr2);
    auto tr3 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq1);
    auto tr4 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {6, -1, 1}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {6, -1, 1}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3}}}), dot);
    auto tr6 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), resh3);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), tr6);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij,ij->ij"}}, mm.get_parameters()));
}

TEST_CASE(einsum_vector_outer_product_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {3}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {5}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), tr2);
    auto tr3 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq1);
    auto tr4 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 1}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 1}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 5}}}), dot);
    auto tr6 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), resh3);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), tr6);

    EXPECT(mm == make_op_module("einsum", {{"equation", "i,j->ij"}}, mm.get_parameters()));
}

TEST_CASE(einsum_matrix_outer_product_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {2, 5}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2, 3}}}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 1}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 1}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 2, 5}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), resh3);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), tr6);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij,kl->ijkl"}}, mm.get_parameters()));
}

TEST_CASE(einsum_batch_matrix_multiplication_op_builder_test)
{
    migraphx::module mm;
    auto a   = mm.add_parameter("a", {migraphx::shape::float_type, {3, 2, 5}});
    auto b   = mm.add_parameter("b", {migraphx::shape::float_type, {3, 5, 3}});
    auto tr1 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), tr1);
    auto tr2 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 5}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 5}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 3, 1}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ijk,ikl->ijl"}}, mm.get_parameters()));
}

TEST_CASE(einsum_tensor_contraction_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3, 5, 7}});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, {1, 3, 3, 7, 5}});
    auto tr1 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4, 5, 6}}}), tr1);
    auto tr2 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 4, 0, 1, 3}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 3}}}), tr2);
    auto tr3   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 5, 6, 1, 2}}}), unsq1);
    auto tr4 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 5, 6, 1, 2}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 15}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 15}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 =
        mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 7, 1, 3, 7, 1, 1}}}), dot);
    auto tr6 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 5, 6, 1, 2, 3, 4}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 2}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3, 4}}}), sq);

    EXPECT(mm ==
           make_op_module("einsum", {{"equation", "pqrs,tuqvr->pstuv"}}, mm.get_parameters()));
}

TEST_CASE(einsum_matrix_diagonal_op_builder_test)
{
    migraphx::module mm;
    auto indices_arg = mm.add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::int64_type, {3, 2}}, {0, 0, 1, 1, 2, 2}});
    auto a           = mm.add_parameter("a", {migraphx::shape::float_type, {3, 3}});
    auto op =
        mm.add_instruction(migraphx::make_op("gathernd", {{"batch_dims", 0}}), a, indices_arg);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), op);
    op = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ii->i"}}, mm.get_parameters()));
}

TEST_CASE(einsum_batch_matrix_diagonal_op_builder_test)
{
    migraphx::module mm;
    auto indices_arg =
        mm.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {3, 3, 2}},
                                         {0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2}});
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {3, 3, 3}});
    auto op =
        mm.add_instruction(migraphx::make_op("gathernd", {{"batch_dims", 1}}), a, indices_arg);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), op);
    op = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "...ii->...i"}}, mm.get_parameters()));
}

TEST_CASE(einsum_3d_diagonal_op_builder_test)
{
    migraphx::module mm;
    auto indices_arg = mm.add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::int64_type, {3, 3}}, {0, 0, 0, 1, 1, 1, 2, 2, 2}});
    auto a           = mm.add_parameter("a", {migraphx::shape::float_type, {3, 3, 3}});
    auto op =
        mm.add_instruction(migraphx::make_op("gathernd", {{"batch_dims", 0}}), a, indices_arg);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), op);
    op = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "iii->i"}}, mm.get_parameters()));
}

TEST_CASE(einsum_diag_vector_multiply_op_builder_test)
{
    migraphx::module mm;

    auto lit   = mm.add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::int64_type, {3, 2}}, {0, 0, 1, 1, 2, 2}});
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {3, 3}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {3}});
    auto gath  = mm.add_instruction(migraphx::make_op("gathernd", {{"batch_dims", 0}}), a, lit);
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), gath);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr2);
    auto tr3   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), unsq1);
    auto tr4   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 1}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 1}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3}}}), dot);
    auto tr6   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), resh3);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), tr6);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ii,i->i"}}, mm.get_parameters()));
}

TEST_CASE(einsum_matrix_trace_op_builder_test)
{
    migraphx::module mm;
    auto indices_arg = mm.add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::int64_type, {3, 2}}, {0, 0, 1, 1, 2, 2}});
    auto a           = mm.add_parameter("a", {migraphx::shape::float_type, {3, 3}});
    auto op =
        mm.add_instruction(migraphx::make_op("gathernd", {{"batch_dims", 0}}), a, indices_arg);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), op);
    op = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), op);
    mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ii->"}}, mm.get_parameters()));
}

TEST_CASE(einsum_matrix_trace_implicit_op_builder_test)
{
    migraphx::module mm;
    auto indices_arg = mm.add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::int64_type, {3, 2}}, {0, 0, 1, 1, 2, 2}});
    auto a           = mm.add_parameter("a", {migraphx::shape::float_type, {3, 3}});
    auto op =
        mm.add_instruction(migraphx::make_op("gathernd", {{"batch_dims", 0}}), a, indices_arg);
    op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), op);
    op = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), op);
    mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ii"}}, mm.get_parameters()));
}

TEST_CASE(einsum_2d_3d_multiplication_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {3, 3}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {3, 4, 5}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2, 3}}}), tr1);
    auto tr2 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4, 5, 1}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij,jkl"}}, mm.get_parameters()));
}

TEST_CASE(einsum_element_wise_multiplication_and_row_sum_op_builder_test)
{
    migraphx::module mm;
    auto a       = mm.add_parameter("a", {migraphx::shape::float_type, {3}});
    auto b       = mm.add_parameter("b", {migraphx::shape::float_type, {3, 4}});
    auto tr1     = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), a);
    auto unsq1   = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), tr1);
    auto tr2     = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), b);
    auto unsq2   = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr2);
    auto red_sum = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), unsq2);
    auto tr3 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), red_sum);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 1}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 1}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 1}}}), dot);
    auto tr6 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), resh3);
    auto sq  = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "i,ij->i"}}, mm.get_parameters()));
}

TEST_CASE(einsum_broadcast_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {3, 1}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {2, 2}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), unsq2);
    auto mbrc =
        mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 1, 2}}}), tr3);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 2}}}), mbrc);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 1}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ij, jk -> ik"}}, mm.get_parameters()));
}

TEST_CASE(einsum_3d_broadcast_op_builder_test)
{
    migraphx::module mm;
    auto a   = mm.add_parameter("a", {migraphx::shape::float_type, {1, 3, 1}});
    auto b   = mm.add_parameter("b", {migraphx::shape::float_type, {2, 2, 4}});
    auto tr1 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), tr1);
    auto tr2 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), unsq2);
    auto mbrc =
        mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 1, 2}}}), tr3);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), mbrc);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 4, 1}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "bik,bkj->bij"}}, mm.get_parameters()));
}

TEST_CASE(einsum_3d_opposite_broadcast_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {1, 3, 2}});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, {2, 1, 4}});

    auto tr1 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), tr1);
    auto tr2 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), unsq2);
    auto mbrc1 =
        mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 1, 2}}}), tr3);
    auto mbrc2 =
        mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1, 4, 2}}}), tr4);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), mbrc1);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), mbrc2);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 4, 1}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "bik,bkj->bij"}}, mm.get_parameters()));
}

TEST_CASE(einsum_3_inputs_op_builder_test)
{
    migraphx::module mm;
    auto a    = mm.add_parameter("a", {migraphx::shape::float_type, {2, 2, 2}});
    auto b    = mm.add_parameter("b", {migraphx::shape::float_type, {2, 2}});
    auto c    = mm.add_parameter("c", {migraphx::shape::float_type, {2, 2, 2}});
    auto tr_1 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), a);
    auto unsq_1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3, 4, 5}}}), tr_1);
    auto reds_1 = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), unsq_1);
    auto tr_2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), b);
    auto unsq_2 =
        mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1, 4, 5}}}), tr_2);
    auto tr_3 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 4, 5, 1, 3}}}), reds_1);
    auto tr_4 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 4, 5, 1, 3}}}), unsq_2);
    auto resh_1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 1}}}), tr_3);
    auto resh_2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 1}}}), tr_4);
    auto tr_5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh_2);
    auto dot_1 = mm.add_instruction(migraphx::make_op("dot"), resh_1, tr_5);
    auto resh_3 =
        mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 1, 1, 2, 2}}}), dot_1);
    auto tr_6 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 4, 1, 5, 2, 3}}}), resh_3);
    auto tr_7 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), c);
    auto unsq_3 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1, 2}}}), tr_7);
    auto reds_2 = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {5}}}), unsq_3);
    auto tr_8   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 5, 1, 2, 4, 3}}}), tr_6);
    auto tr_9 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 5, 1, 2, 4, 3}}}), reds_2);
    auto resh_4 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 2}}}), tr_8);
    auto resh_5 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 2}}}), tr_9);
    auto tr_10 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh_5);
    auto dot_2 = mm.add_instruction(migraphx::make_op("dot"), resh_4, tr_10);
    auto resh_6 =
        mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 2, 2, 1}}}), dot_2);
    auto tr_11 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 5, 4, 1}}}), resh_6);
    auto sq_1 = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 3, 5}}}), tr_11);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), sq_1);

    EXPECT(mm == make_op_module("einsum", {{"equation", "bac,cd,def->ebc"}}, mm.get_parameters()));
}

TEST_CASE(einsum_bilinear_transformation_op_builder_test)
{
    migraphx::module mm;
    auto a      = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto b      = mm.add_parameter("b", {migraphx::shape::float_type, {5, 3, 7}});
    auto c      = mm.add_parameter("c", {migraphx::shape::float_type, {2, 7}});
    auto tr_1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    auto unsq_1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 3}}}), tr_1);
    auto tr_2 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), b);
    auto unsq_2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), tr_2);
    auto tr_3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), unsq_1);
    auto tr_4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), unsq_2);
    auto resh_1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr_3);
    auto resh_2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 3}}}), tr_4);
    auto tr_5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh_2);
    auto dot_1  = mm.add_instruction(migraphx::make_op("dot"), resh_1, tr_5);
    auto resh_3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 5, 7, 1}}}), dot_1);
    auto tr_6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), resh_3);
    auto tr_7   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), c);
    auto unsq_3 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), tr_7);
    auto tr_8 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), tr_6);
    auto tr_9 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), unsq_3);
    auto resh_4 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 7}}}), tr_8);
    auto resh_5 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 7}}}), tr_9);
    auto tr_10 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh_5);
    auto dot_2  = mm.add_instruction(migraphx::make_op("dot"), resh_4, tr_10);
    auto resh_6 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1, 5, 1}}}), dot_2);
    auto tr_11 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), resh_6);
    auto sq_1 = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), tr_11);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), sq_1);

    EXPECT(mm == make_op_module("einsum", {{"equation", "ik,jkl,il->ij"}}, mm.get_parameters()));
}

TEST_CASE(einsum_ellipsis_op_builder_test)
{
    migraphx::module mm;
    auto a   = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3, 2}});
    auto b   = mm.add_parameter("b", {migraphx::shape::float_type, {2, 4, 2}});
    auto tr1 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), tr1);
    auto tr2 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 0, 1, 2}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 0, 1, 2}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 4, 1}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 3, 0}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), sq);

    EXPECT(mm ==
           make_op_module("einsum", {{"equation", "...ik,kj...->ij..."}}, mm.get_parameters()));
}

TEST_CASE(einsum_ellipsis_multidim_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {3, 2, 3, 2}});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, {2, 4, 3, 2}});
    auto tr1 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), tr1);
    auto tr2 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2, 3}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), tr2);
    auto tr3   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {3, 4, 0, 1, 2}}}), unsq1);
    auto tr4 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {3, 4, 0, 1, 2}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {6, -1, 2}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {6, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 3, 4, 1}}}), dot);
    auto tr6   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {2, 3, 4, 0, 1}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), sq);

    EXPECT(mm ==
           make_op_module("einsum", {{"equation", "...ik,kj...->ij..."}}, mm.get_parameters()));
}

TEST_CASE(einsum_ellipsis_zero_op_builder_test)
{
    migraphx::module mm;
    auto a   = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3, 2}});
    auto b   = mm.add_parameter("b", {migraphx::shape::float_type, {4, 3, 2}});
    auto tr1 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 1, 0}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), tr1);
    auto tr2 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 1, 0}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 3, 2, 0}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 3, 2, 0}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 2}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 4, 1}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 0, 2, 1}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), sq);

    EXPECT(mm ==
           make_op_module("einsum", {{"equation", "...qhd,...khd->...hqk"}}, mm.get_parameters()));
}

TEST_CASE(einsum_ellipsis_implicit_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {3, 2, 3, 2}});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, {3, 4, 3, 2}});
    auto tr1 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 1, 0}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), tr1);
    auto tr2 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 1, 0}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), tr2);
    auto tr3   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {4, 3, 2, 0, 1}}}), unsq1);
    auto tr4 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {4, 3, 2, 0, 1}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 6}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, -1, 6}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 4, 1, 1}}}), dot);
    auto tr6   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {3, 4, 2, 1, 0}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "...qhd,...khd"}}, mm.get_parameters()));
}

TEST_CASE(einsum_ellipsis_scalar_multiplication_op_builder_test)
{
    migraphx::module mm;
    auto a     = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3}});
    auto b     = mm.add_parameter("b", {migraphx::shape::float_type, {2, 3}});
    auto tr1   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr1);
    auto tr2   = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {}), tr2);
    auto tr3 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq1);
    auto tr4 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {6, -1, 1}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {6, -1, 1}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3}}}), dot);
    auto tr6 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), resh3);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1}}}), tr6);

    EXPECT(mm == make_op_module("einsum", {{"equation", "..., ..."}}, mm.get_parameters()));
}

TEST_CASE(einsum_common_1_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {2, 2, 2, 2}});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, {2, 2, 2, 2}});
    auto tr1 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 2, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4}}}), tr1);
    auto tr2 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 2, 1}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), tr2);
    auto tr3   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 4, 1}}}), unsq1);
    auto tr4 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 4, 1}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1, 2}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 1}}}), dot);
    auto tr6   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 4, 1, 2, 3}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "bsnh,btnh->bnts"}}, mm.get_parameters()));
}

TEST_CASE(einsum_common_2_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {2, 2, 2, 2}});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, {2, 2, 2, 2}});
    auto tr1 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 2, 1}}}), a);
    auto unsq1    = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 5}}}), tr1);
    auto red_sum1 = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), unsq1);
    auto tr2 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 2, 1}}}), b);
    auto unsq2    = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 4}}}), tr2);
    auto red_sum2 = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), unsq2);
    auto tr3      = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 4, 5, 2}}}), red_sum1);
    auto tr4 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 4, 5, 2}}}), red_sum2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 =
        mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 2, 2, 1}}}), dot);
    auto tr6 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 5, 2, 3, 4}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1, 2}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "bsnh,ctnh->nts"}}, mm.get_parameters()));
}

TEST_CASE(einsum_common_3_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {2, 2, 2, 2}});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, {2, 2, 2, 2}});
    auto tr1 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), a);
    auto unsq1    = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), tr1);
    auto red_sum1 = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), unsq1);
    auto tr2 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), b);
    auto unsq2    = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 3}}}), tr2);
    auto red_sum2 = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), unsq2);
    auto tr3      = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 3, 2, 5}}}), red_sum1);
    auto tr4 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 3, 2, 5}}}), red_sum2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 =
        mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 2, 2, 1}}}), dot);
    auto tr6 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 3, 2, 5}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1, 5}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "bnst,chst->shn"}}, mm.get_parameters()));
}

TEST_CASE(einsum_common_4_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {2, 2, 3, 2}});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, {2, 2, 4, 2}});
    auto tr1 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4}}}), tr1);
    auto tr2 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), tr2);
    auto tr3   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 4, 2}}}), unsq1);
    auto tr4 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 4, 2}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1, 2}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 3, 4, 1}}}), dot);
    auto tr6   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 3}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), sq);

    EXPECT(mm == make_op_module("einsum", {{"equation", "bcxd,bcyd->bcxy"}}, mm.get_parameters()));
}

TEST_CASE(einsum_common_5_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {3, 2, 3, 2}});
    auto b = mm.add_parameter("b", {migraphx::shape::float_type, {3, 4, 3, 2}});
    auto tr1 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 1, 0}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), tr1);
    auto tr2 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 2, 1, 0}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), tr2);
    auto tr3   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {1, 4, 3, 2, 0}}}), unsq1);
    auto tr4 = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {1, 4, 3, 2, 0}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {9, -1, 2}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {9, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 3, 2, 4, 1}}}), dot);
    auto tr6   = mm.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {4, 0, 3, 2, 1}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 0, 2, 1}}}), sq);

    EXPECT(mm ==
           make_op_module("einsum", {{"equation", "...qhd,...khd->...hqk"}}, mm.get_parameters()));
}

TEST_CASE(einsum_common_6_op_builder_test)
{
    migraphx::module mm;
    auto a   = mm.add_parameter("a", {migraphx::shape::float_type, {3, 2, 2}});
    auto b   = mm.add_parameter("b", {migraphx::shape::float_type, {2, 2, 3}});
    auto tr1 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), a);
    auto unsq1 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), tr1);
    auto tr2 = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), b);
    auto unsq2 = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), tr2);
    auto tr3 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 0, 1, 2}}}), unsq1);
    auto tr4 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {3, 0, 1, 2}}}), unsq2);
    auto resh1 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), tr3);
    auto resh2 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, -1, 2}}}), tr4);
    auto tr5 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), resh2);
    auto dot   = mm.add_instruction(migraphx::make_op("dot"), resh1, tr5);
    auto resh3 = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 3, 1}}}), dot);
    auto tr6 =
        mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 3, 0}}}), resh3);
    auto sq = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), tr6);
    mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), sq);

    EXPECT(mm ==
           make_op_module("einsum", {{"equation", "i...k,k...j->i...j"}}, mm.get_parameters()));
}

TEST_CASE(einsum_common_7_op_builder_test)
{
    migraphx::module mm;
    auto a  = mm.add_parameter("a", {migraphx::shape::float_type, {5, 5}});
    auto op = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), a);
    op      = mm.add_instruction(migraphx::make_op("unsqueeze", {}), op);
    op      = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), op);
    op      = mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), op);
    op      = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {0}}}), op);

    EXPECT(mm == make_op_module("einsum", {{"equation", "...j->..."}}, mm.get_parameters()));
}
