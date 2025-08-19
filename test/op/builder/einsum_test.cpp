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

bool test_invalid_input(const std::string& equation             = "",
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
