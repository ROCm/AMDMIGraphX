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

TEST_CASE(nonzero_test_nonstd_shape)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 2}};
    std::vector<float> data = {1.0f, 1.3f,   0.0f,  -1.2f, 0.0f, -100.f, 200.f, 0.0f,
                               0.1f, 0.2f,   0.0f,  0.5f,  0.5f, 0.0f,   0.0f,  1.2f,
                               0.0f, -100.f, 200.f, 0.0f,  0.1f, 0.0f,   0.0f,  1.5f};
    auto input              = mm->add_literal(migraphx::literal(s, data));
    auto input_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 3, 0}}}), input);
    auto ret = mm->add_instruction(migraphx::make_op("nonzero"), input_trans);
    mm->add_return({ret});
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    auto gold   = p_uncompiled.eval({}).back();
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold_vector;
    gold.visit([&](auto output) { gold_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(result_vector, gold_vector));
}

TEST_CASE(roialign_test_nonstd_shape)
{
    const std::string& trans_mode   = "half_pixel";
    const std::string& pooling_mode = "avg";
    int64_t sampling_ratio          = 2;
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape x_s{migraphx::shape::float_type, {1, 1, 10, 10}};
    std::vector<float> x_vec = {
        0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856, 0.7250,
        0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324, 0.8992, 0.4467,
        0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766, 0.4308, 0.3400, 0.2162,
        0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406, 0.7724, 0.3921, 0.2541, 0.5799,
        0.4062, 0.2194, 0.4473, 0.4687, 0.7109, 0.9327, 0.9815, 0.6320, 0.1728, 0.6119,
        0.3097, 0.1283, 0.4984, 0.5068, 0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,
        0.1011, 0.8477, 0.4726, 0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689,
        0.1366, 0.3671, 0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928,
        0.5697, 0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
        0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482, 0.0502};

    migraphx::shape roi_s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> roi_vec = {0, 0, 9, 9, 0, 5, 4, 9, 5, 5, 9, 9};

    migraphx::shape ind_s{migraphx::shape::int64_type, {3}};
    std::vector<int64_t> ind_vec = {0, 0, 0};

    auto x = mm->add_literal(migraphx::literal(x_s, x_vec));
    auto xt =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto roi = mm->add_literal(migraphx::literal(roi_s, roi_vec));
    auto ind = mm->add_literal(migraphx::literal(ind_s, ind_vec));
    mm->add_instruction(migraphx::make_op("roialign",
                                          {{"coordinate_transformation_mode", trans_mode},
                                           {"spatial_scale", 1.0},
                                           {"output_height", 5},
                                           {"output_width", 5},
                                           {"sampling_ratio", sampling_ratio},
                                           {"mode", pooling_mode}}),
                        xt,
                        roi,
                        ind);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    auto gold   = p_uncompiled.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold_vector;
    gold.visit([&](auto output) { gold_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, gold_vector));
}

TEST_CASE(scatter_test_nonstd_shape)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
    std::vector<float> vd(sd.elements(), 0.0f);

    migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
    std::vector<int> vi = {1, 0, 2, 0, 2, 1};

    migraphx::shape su{migraphx::shape::float_type, {2, 3}};
    std::vector<float> vu = {1.0, 1.1, 1.2, 2.0, 2.1, 2.2};

    auto ld  = mm->add_literal(migraphx::literal{sd, vd});
    auto li  = mm->add_literal(migraphx::literal{si, vi});
    auto lu  = mm->add_literal(migraphx::literal{su, vu});
    auto ldt = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), ld);
    auto r   = mm->add_instruction(migraphx::make_op("scatter", {{"axis", 0}}), ldt, li, lu);
    mm->add_return({r});
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    auto gold = p_uncompiled.eval({}).back();
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold_vector;
    gold.visit([&](auto output) { gold_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, gold_vector));
}

TEST_CASE(squeeze_transpose_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(4 * 3 * 3);
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s1, data});
    auto l0_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 3, 0, 4}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze"), l0_trans);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result          = p.eval({}).back();
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    migraphx::run_passes(*mm_uncompiled, {migraphx::auto_contiguous{}}); 
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == expected_result.get_shape());
}

TEST_CASE(squeeze_multibroadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(3 * 3);
    migraphx::shape s1{migraphx::shape::float_type, {1, 3, 1, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s1, data});
    auto l0_brcst =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 3, 4, 3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze"), l0_brcst);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result          = p.eval({}).back();
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    migraphx::run_passes(*mm_uncompiled, {migraphx::auto_contiguous{}}); 
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == expected_result.get_shape());
}

TEST_CASE(squeeze_slice_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(2 * 3 * 4 * 4);
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto l0 = mm->add_literal(migraphx::literal{s1, data});
    auto l0_slice =
        mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze"), l0_slice);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result          = p.eval({}).back();
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    migraphx::run_passes(*mm_uncompiled, {migraphx::auto_contiguous{}}); 
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == expected_result.get_shape());
}

TEST_CASE(unsqueeze_nonstd_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(4 * 3 * 3);
    migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s1, data});
    auto l0_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), l0);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l0_trans);
    auto p_uncompiled = p;
    p.compile(migraphx::ref::target{});
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == expected_result.get_shape());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
