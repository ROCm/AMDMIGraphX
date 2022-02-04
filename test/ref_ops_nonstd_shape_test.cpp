#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/pass_manager.hpp>
#include "test.hpp"

TEST_CASE(argmax_test_nonstd_shape)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    auto dl_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), dl);
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", -3}}), dl_trans);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result   = p.eval({}).back();
    auto res_gold = p_uncompiled.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<int64_t> res_gold_vec;
    res_gold.visit([&](auto output) { res_gold_vec.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(result_vec, res_gold_vec));
}

TEST_CASE(argmin_test_nonstd_shape)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    auto dl_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), dl);
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", -1}}), dl_trans);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result   = p.eval({}).back();
    auto res_gold = p_uncompiled.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<int64_t> res_gold_vec;
    res_gold.visit([&](auto output) { res_gold_vec.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(result_vec, res_gold_vec));
}

TEST_CASE(squeeze_transpose_test)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {
        1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545,  2.5758,  -1.6849,
        0.0928,  0.9022,  -0.8765, -0.4090, 0.9301,  2.0724, -1.5706, 0.4867,  -0.1493,
        0.6957,  -0.2179, 0.7142,  0.7177,  0.0183,  1.3497, 1.2255,  1.6834,  -2.0305,
        -0.3221, 1.2255,  1.6834,  -2.0305, -0.3221, 1.2255, 1.6834,  -2.0305, -0.3221};
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s1, data});
    auto l0_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 3, 0, 4}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze"), l0_trans);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    // contiguous is required to read the values in standard shaped order
    auto tr_op               = migraphx::make_op("contiguous");
    auto std_expected_result = tr_op.compute(result.get_shape(), {expected_result});
    EXPECT(result == std_expected_result);
}

TEST_CASE(squeeze_multibroadcast_test)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {
        1.2255, 1.6834, -2.0305, -0.3221, 0.4701, 0.2583, 0.7545, 2.5758, -1.6849};
    migraphx::shape s1{migraphx::shape::float_type, {1, 3, 1, 3}};
    auto l0       = mm->add_literal(migraphx::literal{s1, data});
    auto l0_brcst = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 3, 4, 3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze"), l0_brcst);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result              = p.eval({}).back();
    auto expected_result     = p_uncompiled.eval({}).back();
    auto tr_op               = migraphx::make_op("contiguous");
    auto std_expected_result = tr_op.compute(result.get_shape(), {expected_result});
    EXPECT(result == std_expected_result);
}

TEST_CASE(squeeze_slice_test)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {
        1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545,  2.5758,  -1.6849,
        0.0928,  0.9022,  -0.8765, -0.4090, 0.9301,  2.0724, -1.5706, 0.4867,  -0.1493,
        0.6957,  -0.2179, 0.7142,  0.7177,  0.0183,  1.3497, 1.2255,  1.6834,  -2.0305,
        -0.3221, 1.2255,  1.6834,  -2.0305, -0.3221, 1.2255, 1.6834,  -2.0305, -0.3221};
    migraphx::shape s1{migraphx::shape::float_type, {1, 3, 4, 3}};
    auto l0       = mm->add_literal(migraphx::literal{s1, data});
    auto l0_slice = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze"), l0_slice);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result              = p.eval({}).back();
    auto expected_result     = p_uncompiled.eval({}).back();
    auto tr_op               = migraphx::make_op("contiguous");
    auto std_expected_result = tr_op.compute(result.get_shape(), {expected_result});
    EXPECT(result == std_expected_result);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
