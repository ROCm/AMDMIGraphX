/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include "test.hpp"

TEST_CASE(argmax_test_nonstd_shape)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto dl = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}));
    auto dl_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), dl);
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", -3}}), dl_trans);
    auto p_uncompiled = p;
    p.compile(migraphx::make_target("ref"));
    auto result   = p.eval({}).back();
    auto res_gold = p_uncompiled.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<int64_t> res_gold_vec;
    res_gold.visit([&](auto output) { res_gold_vec.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(result_vec, res_gold_vec));
}

TEST_CASE(argmin_test_nonstd_shape)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto dl = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}));
    auto dl_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), dl);
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", -1}}), dl_trans);
    auto p_uncompiled = p;
    p.compile(migraphx::make_target("ref"));
    auto result   = p.eval({}).back();
    auto res_gold = p_uncompiled.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<int64_t> res_gold_vec;
    res_gold.visit([&](auto output) { res_gold_vec.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(result_vec, res_gold_vec));
}

TEST_CASE(isnan_broadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {3}};
    migraphx::shape s1{migraphx::shape::float_type, {3, 2}};
    auto nan_val             = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> data0 = {1.2, 5.2, nan_val};
    auto l0                  = mm->add_literal(migraphx::literal{s0, data0});
    auto l1                  = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", s1.lens()}}), l0);
    mm->add_instruction(migraphx::make_op("isnan"), l1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> correct = {0, 0, 0, 0, 1, 1};
    EXPECT(migraphx::verify::verify_range(results_vector, correct));
}

TEST_CASE(squeeze_transpose_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 1, 3, 1, 3}}));
    auto l0_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 3, 0, 4}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze"), l0_trans);
    auto p_uncompiled = p;
    // contiguous is required to read the values in standard shaped order
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    mm_uncompiled->add_instruction(migraphx::make_op("contiguous"),
                                   std::prev(mm_uncompiled->end()));
    p.compile(migraphx::make_target("ref"));
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {3, 4, 3}});
    EXPECT(result == expected_result);
}

TEST_CASE(squeeze_multibroadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 3, 1, 3}}));
    auto l0_brcst = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 3, 4, 3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze"), l0_brcst);
    auto p_uncompiled   = p;
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    mm_uncompiled->add_instruction(migraphx::make_op("contiguous"),
                                   std::prev(mm_uncompiled->end()));
    p.compile(migraphx::make_target("ref"));
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {4, 3, 4, 3}});
    EXPECT(result == expected_result);
}

TEST_CASE(squeeze_slice_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 3, 4, 3}}));
    auto l0_slice = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze"), l0_slice);
    auto p_uncompiled   = p;
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    mm_uncompiled->add_instruction(migraphx::make_op("contiguous"),
                                   std::prev(mm_uncompiled->end()));
    p.compile(migraphx::make_target("ref"));
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {3, 3}});
    EXPECT(result == expected_result);
}

TEST_CASE(unsqueeze_transpose_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
    auto l0 = mm->add_literal(migraphx::generate_literal(s1));
    auto l0_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), l0);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l0_trans);
    auto p_uncompiled   = p;
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    mm_uncompiled->add_instruction(migraphx::make_op("contiguous"),
                                   std::prev(mm_uncompiled->end()));
    p.compile(migraphx::make_target("ref"));
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {3, 4, 1, 3}});
    EXPECT(result == expected_result);
}

TEST_CASE(unsqueeze_multibroadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3}};
    auto l0 = mm->add_literal(migraphx::generate_literal(s1));
    auto l0_brcst =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4, 3, 3}}}), l0);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l0_brcst);
    auto p_uncompiled   = p;
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    mm_uncompiled->add_instruction(migraphx::make_op("contiguous"),
                                   std::prev(mm_uncompiled->end()));
    p.compile(migraphx::make_target("ref"));
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {4, 4, 1, 3, 3}});
    EXPECT(result == expected_result);
}

TEST_CASE(unsqueeze_slice_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto l0       = mm->add_literal(migraphx::generate_literal(s1));
    auto l0_slice = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {3}}, {"starts", {2}}, {"ends", {3}}}), l0);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l0_slice);
    auto p_uncompiled   = p;
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    mm_uncompiled->add_instruction(migraphx::make_op("contiguous"),
                                   std::prev(mm_uncompiled->end()));
    p.compile(migraphx::make_target("ref"));
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {2, 1, 3, 4, 1}});
    EXPECT(result == expected_result);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
