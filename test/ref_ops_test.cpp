#include <iostream>
#include <vector>
#include <cmath>
#include <migraphx/literal.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/batch_norm_inference.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include "test.hpp"
#include <migraphx/half.hpp>
#include <iomanip>

float sigmoid(float x) { return 1 / (1 + expf(-x)); }

float elu(float a, float x) { return x > 0 ? x : a * std::expm1(x); }

TEST_CASE(abs_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l = mm->add_literal(migraphx::literal{s, {-1, 2, -3, 4}});
    mm->add_instruction(migraphx::make_op("abs"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 2, 3, 4};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(acos_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{-0.8f, 0.0f, 1.0f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("acos"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return acosf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(acosh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{1.1f, 1.2f, 2.0f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("acosh"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return acoshf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(add_broadcast_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3}};
        std::vector<float> a_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        migraphx::shape b_shape{migraphx::shape::float_type, {2, 2}};
        std::vector<float> b_data{0, -1, -2, -3};
        uint64_t axis = 0;
        auto l1       = mm->add_literal(migraphx::literal{a_shape, a_data});
        auto l2       = mm->add_literal(migraphx::literal{b_shape, b_data});
        auto l3       = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l1->get_shape().lens()}}),
            l2);
        mm->add_instruction(migraphx::make_op("add"), l1, l3);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        EXPECT(result.get_shape().packed());
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3}};
        std::vector<float> a_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        migraphx::shape b_shape{migraphx::shape::float_type, {2, 2, 1}};
        std::vector<float> b_data{0, -1, -2, -3};
        auto l1 = mm->add_literal(migraphx::literal{a_shape, a_data});
        auto l2 = mm->add_literal(migraphx::literal{b_shape, b_data});
        auto l3 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3}}}), l1);
        auto l4 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3}}}), l2);
        mm->add_instruction(migraphx::make_op("add"), l3, l4);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        EXPECT(result.get_shape().packed());
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
}

TEST_CASE(add_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    mm->add_instruction(migraphx::make_op("add"), l1, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 2, 4};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(argmax_test_0)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", 0}}), dl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vec, res_gold));
}

TEST_CASE(argmax_test_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {0, 0, 2, 1, 2, 0, 0, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", 1}}), dl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vec, res_gold));
}

TEST_CASE(argmax_test_2)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {1, 3, 2, 2, 2, 3};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), dl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vec, res_gold));
}

TEST_CASE(argmax_test_neg_2)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {0, 0, 2, 1, 2, 0, 0, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", -2}}), dl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_0)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 0}}), dl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {2, 2, 0, 2, 0, 1, 2, 0};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), dl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_2)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {2, 1, 0, 3, 3, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 2}}), dl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_neg_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {2, 1, 0, 3, 3, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", -1}}), dl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vec, res_gold));
}

TEST_CASE(asin_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-0.5f, 0.0f, 0.9f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("asin"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return asinf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(asinh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-0.5f, 0.0f, 0.9f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("asinh"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return asinhf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(atan_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{-1.0f, 0.0f, 1.0f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("atan"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return atanf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(atanh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{0.4435683f, 0.6223626f, 0.316958f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("atanh"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return atanhf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(avgpool_test)
{
    // 1D case 1, input is 3D
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
        auto op    = migraphx::op::pooling{"average"};
        op.lengths = {2};
        op.padding = {0};
        op.stride  = {1};

        std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(op, l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();

        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0.25, 0.3, 0.25, 0.65, 0.7, 0.5, 0.4, 0.4, 0.35};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }

    // 1D case 2, stride 2
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {2, 2, 4}};
        auto op    = migraphx::op::pooling{"average"};
        op.lengths = {2};
        op.padding = {1};
        op.stride  = {2};

        std::vector<float> data{1.6321,
                                -2.4186,
                                0.2239,
                                -1.4232,
                                0.8158,
                                0.4103,
                                -0.3149,
                                -0.1361,
                                -0.3442,
                                2.007,
                                0.4331,
                                1.5295,
                                0.9965,
                                0.4766,
                                1.0942,
                                -0.2915};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(op, l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.6321,
                                -1.0974,
                                -1.4232,
                                0.8158,
                                0.0477,
                                -0.1361,
                                -0.3442,
                                1.22005,
                                1.5295,
                                0.9965,
                                0.7854,
                                -0.2915};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }

    // 3D, input is 5D
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 3, 3}};
        auto op    = migraphx::op::pooling{"average"};
        op.lengths = {2, 2, 2};
        op.padding = {0, 0, 0};
        op.stride  = {1, 1, 1};

        std::vector<float> data{
            -0.179, -1.756, 0.651,  1.955,  1.87,   -0.604, 0.247,  0.449,  -0.137, 1.187,  1.593,
            0.424,  2.698,  -0.104, -0.069, -1.293, 0.538,  1.291,  0.974,  1.096,  0.74,   -0.669,
            -1.08,  -1.041, -1.407, 1.43,   -0.211, -0.017, 0.532,  1.276,  0.627,  0.236,  -0.396,
            -0.204, 0.501,  -0.599, -1.414, -0.615, -0.274, 0.168,  -0.144, 0.5,    1.42,   1.082,
            -0.952, -0.846, -1.244, 1.475,  1.246,  1.344,  -1.722, -1.24,  -0.851, 0.06,   0.507,
            0.762,  -0.007, -1.484, 1.028,  0.317,  1.077,  -1.289, 0.875,  -0.417, -0.673, 1.715,
            -0.307, 0.264,  -0.973, 1.412,  2.561,  -0.515, -0.201, 0.827,  -1.231, 1.958,  -0.552,
            0.036,  -0.993, -0.859, -1.458, -0.575, 0.048,  -0.779, -1.025, -1.135, 1.166,  -0.131,
            0.726,  0.52,   0.467,  -0.494, 0.675,  0.203,  -0.63,  -0.918, -0.5,   -1.395, 1.39,
            1.705,  0.444,  -0.835, -0.506, 0.101,  0.602,  0.543,  0.357,  1.042};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(op, l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{
            0.908,     0.250625,  0.795,     0.40425, 0.711875,  0.194875,  0.014125,  0.09425,
            -0.078375, 0.139375,  0.46075,   0.0285,  -0.188125, -0.085,    0.378125,  -0.085375,
            -0.04,     0.304125,  0.40775,   0.2835,  0.112375,  -0.073375, 0.4355,    -0.187,
            -0.392625, -0.258375, -0.485875, -0.0345, 0.16125,   -0.131875, -0.228375, 0.068625};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
}

TEST_CASE(batch_norm_1d_per_actv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape x_shape{migraphx::shape::float_type, {2, 2, 4}};
    migraphx::shape c_shape(migraphx::shape::float_type, {2, 4});

    std::vector<float> x_data     = {0.3547,
                                 0.477,
                                 -1.8575,
                                 0.663,
                                 -0.1881,
                                 -0.5113,
                                 -0.1803,
                                 -0.5915,
                                 -0.1552,
                                 0.9821,
                                 1.827,
                                 0.0558,
                                 -0.0417,
                                 -1.0693,
                                 1.9948,
                                 -0.7448};
    std::vector<float> scale_data = {
        -0.3181, -0.3885, 1.655, 0.0704, -0.2565, -1.1761, -0.3751, 0.1057};
    std::vector<float> bias_data = {
        -1.2118, -2.1156, 0.0046, -0.1341, -0.2724, -1.0718, 0.5535, -0.889};
    std::vector<float> mean_data = {
        0.0997, 0.7295, -0.0153, 0.3594, -0.1149, -0.7903, 0.9073, -0.6681};
    std::vector<float> variance_data = {
        0.13, 0.1276, 6.7878, 0.1843, 0.0107, 0.1556, 2.3655, 0.0117};

    auto x        = mm->add_literal(migraphx::literal{x_shape, x_data});
    auto scale    = mm->add_literal(migraphx::literal{c_shape, scale_data});
    auto bias     = mm->add_literal(migraphx::literal{c_shape, bias_data});
    auto mean     = mm->add_literal(migraphx::literal{c_shape, mean_data});
    auto variance = mm->add_literal(migraphx::literal{c_shape, variance_data});

    mm->add_instruction(
        migraphx::make_op(
            "batch_norm_inference",
            {{"epsilon", 1e-6},
             {"momentum", 0.9},
             {"bn_mode", migraphx::to_value(migraphx::op::batch_norm_inference::per_activation)}}),
        x,
        scale,
        bias,
        mean,
        variance);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-1.43677,
                               -1.84098,
                               -1.16563,
                               -0.0843136,
                               -0.090896,
                               -1.90364,
                               0.81875,
                               -0.81415,
                               -0.986915,
                               -2.39032,
                               1.17489,
                               -0.183886,
                               -0.453904,
                               -0.239955,
                               0.288275,
                               -0.963948};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(batch_norm_1d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape c_shape(migraphx::shape::float_type, {3});

    std::vector<float> x_data        = {0.7253,  -0.6356, 0.4606,  -0.8689, -1.1932, 0.4538,
                                 -1.0018, -0.365,  -0.214,  -0.9553, -0.7672, 0.2331,
                                 -0.8416, -0.6142, 0.0814,  0.2498,  -0.6706, 1.4872,
                                 0.5112,  -1.5212, -0.9126, 0.0735,  1.085,   -0.3417};
    std::vector<float> scale_data    = {1.1, 1.2, 1.3};
    std::vector<float> bias_data     = {0.1, 0.2, 0.3};
    std::vector<float> mean_data     = {-0.1804, -0.2875, -0.2249};
    std::vector<float> variance_data = {2.7914, 7.3424, 3.3287};

    auto x        = mm->add_literal(migraphx::literal{x_shape, x_data});
    auto scale    = mm->add_literal(migraphx::literal{c_shape, scale_data});
    auto bias     = mm->add_literal(migraphx::literal{c_shape, bias_data});
    auto mean     = mm->add_literal(migraphx::literal{c_shape, mean_data});
    auto variance = mm->add_literal(migraphx::literal{c_shape, variance_data});

    mm->add_instruction(migraphx::make_op("batch_norm_inference", {{"epsilon", 1e-5}}),
                        x,
                        scale,
                        bias,
                        mean,
                        variance);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.696301,  -0.199697, 0.522026,  -0.353299, -0.201094, 0.528289,
                               -0.116332, 0.165679,  0.307767,  -0.220435, -0.086407, 0.62634,
                               -0.335325, -0.185608, 0.272366,  0.383238,  0.0303421, 0.985936,
                               0.553709,  -0.346351, -0.190009, 0.51262,   1.23335,   0.216776};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(batch_norm_3d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape x_shape{migraphx::shape::float_type, {2, 2, 2, 2, 2}};
    migraphx::shape c_shape(migraphx::shape::float_type, {2});

    std::vector<float> x_data     = {-1.0833, 1.9681,  1.2075,  -0.723,  -0.4076, -0.8738, 0.5853,
                                 -0.5357, 1.734,   0.7904,  0.6953,  -0.468,  -0.425,  0.6895,
                                 0.0096,  0.4205,  -0.1749, 1.2821,  2.1453,  -0.8538, 1.0687,
                                 0.0906,  0.0714,  -1.3079, -0.6376, 1.3023,  0.945,   0.0927,
                                 -0.7421, -1.4341, -1.0309, 1.5153};
    std::vector<float> scale_data = {1.1, 1.3};
    std::vector<float> bias_data  = {0.1, 0.2};
    std::vector<float> mean_data  = {0.1537, 0.2161};
    std::vector<float> variance_data = {18.0805, 13.3906};

    auto x        = mm->add_literal(migraphx::literal{x_shape, x_data});
    auto scale    = mm->add_literal(migraphx::literal{c_shape, scale_data});
    auto bias     = mm->add_literal(migraphx::literal{c_shape, bias_data});
    auto mean     = mm->add_literal(migraphx::literal{c_shape, mean_data});
    auto variance = mm->add_literal(migraphx::literal{c_shape, variance_data});

    mm->add_instruction(migraphx::make_op("batch_norm_inference"), x, scale, bias, mean, variance);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        -0.220005, 0.569376, 0.372612, -0.126798,  -0.0452053, -0.165809, 0.211653,  -0.0783441,
        0.739245,  0.404024, 0.370239, -0.0430317, -0.0277556, 0.368179,  0.126639,  0.272615,
        0.0149929, 0.391911, 0.615216, -0.160635,  0.336706,   0.0836764, 0.0787094, -0.278108,
        -0.103283, 0.585881, 0.458947, 0.156161,   -0.140408,  -0.386246, -0.243006, 0.661551};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(batch_norm_inference_test)
{
    migraphx::program p;
    auto* mm                 = p.get_main_module();
    const size_t width       = 2;
    const size_t height      = 2;
    const size_t channels    = 4;
    const size_t batches     = 2;
    const float x_val        = 8.0;
    const float mean_val     = 2.0;
    const float variance_val = 4.0;
    const float scale_val    = 2.0f;
    const float bias_val     = 1.0f;
    const float output_val = scale_val * (x_val - mean_val) / (std::sqrt(variance_val)) + bias_val;

    migraphx::shape s{migraphx::shape::float_type, {batches, channels, height, width}};
    migraphx::shape vars{migraphx::shape::float_type, {channels}};
    std::vector<float> x_data(width * height * channels * batches);
    std::vector<float> scale_data(channels);
    std::vector<float> bias_data(channels);
    std::vector<float> mean_data(channels);
    std::vector<float> variance_data(channels);

    std::fill(x_data.begin(), x_data.end(), x_val);
    std::fill(mean_data.begin(), mean_data.end(), mean_val);
    std::fill(variance_data.begin(), variance_data.end(), variance_val);
    std::fill(scale_data.begin(), scale_data.end(), scale_val);
    std::fill(bias_data.begin(), bias_data.end(), bias_val);

    auto x        = mm->add_literal(migraphx::literal{s, x_data});
    auto scale    = mm->add_literal(migraphx::literal{vars, scale_data});
    auto bias     = mm->add_literal(migraphx::literal{vars, bias_data});
    auto mean     = mm->add_literal(migraphx::literal{vars, mean_data});
    auto variance = mm->add_literal(migraphx::literal{vars, variance_data});

    mm->add_instruction(migraphx::make_op("batch_norm_inference"), x, scale, bias, mean, variance);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> result_vector(width * height * channels * batches);
    std::vector<float> gold(width * height * channels * batches);
    std::fill(gold.begin(), gold.end(), output_val);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(broadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    uint64_t axis = 0;
    auto l1       = mm->add_literal(migraphx::literal{a_shape, a_data});
    auto l2       = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l1->get_shape().lens()}}), l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    auto output = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -2);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(ceil_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    std::vector<float> data = {1.1, 1.5, 1.6, -1.1, -1.5, -1.6, 0.0, 2.0, -2.0};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("ceil"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return std::ceil(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(clip_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l       = mm->add_literal(migraphx::literal{s, {-1.0, 0.0, 10.0}});
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), min_val);
    max_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), max_val);
    mm->add_instruction(migraphx::make_op("clip"), l, min_val, max_val);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.0, 0.0, 6.0};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(concat_test)
{
    {
        migraphx::program p;
        auto* mm               = p.get_main_module();
        int axis               = 1;
        std::vector<int> data0 = {0, 1, 5, 6};
        std::vector<int> data1 = {2, 3, 4, 7, 8, 9};
        std::vector<int> data2 = {10, 20};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::int32_type, {2, 1}};
        auto l0 = mm->add_literal(migraphx::literal{s0, data0});
        auto l1 = mm->add_literal(migraphx::literal{s1, data1});
        auto l2 = mm->add_literal(migraphx::literal{s2, data2});
        mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
        p.compile(migraphx::ref::target{});
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9, 20};
        std::vector<int> results_vector(2 * 6);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(migraphx::verify_range(result.get_shape().lens(), std::vector<std::size_t>({2, 6})));
        EXPECT(
            migraphx::verify_range(result.get_shape().strides(), std::vector<std::size_t>({6, 1})));
    }

    {
        migraphx::program p;
        auto* mm               = p.get_main_module();
        int axis               = -1;
        std::vector<int> data0 = {0, 1, 5, 6};
        std::vector<int> data1 = {2, 3, 4, 7, 8, 9};
        std::vector<int> data2 = {10, 20};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::int32_type, {2, 1}};
        auto l0 = mm->add_literal(migraphx::literal{s0, data0});
        auto l1 = mm->add_literal(migraphx::literal{s1, data1});
        auto l2 = mm->add_literal(migraphx::literal{s2, data2});
        mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
        p.compile(migraphx::ref::target{});
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9, 20};
        std::vector<int> results_vector(2 * 6);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(migraphx::verify_range(result.get_shape().lens(), std::vector<std::size_t>({2, 6})));
        EXPECT(
            migraphx::verify_range(result.get_shape().strides(), std::vector<std::size_t>({6, 1})));
    }

    {
        migraphx::program p;
        auto* mm               = p.get_main_module();
        int axis               = 0;
        std::vector<int> data0 = {0, 1, 2, 3};
        std::vector<int> data1 = {4, 5, 6, 7, 8, 9};
        std::vector<int> data2 = {10, 11};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::int32_type, {1, 2}};
        auto l0 = mm->add_literal(migraphx::literal{s0, data0});
        auto l1 = mm->add_literal(migraphx::literal{s1, data1});
        auto l2 = mm->add_literal(migraphx::literal{s2, data2});
        mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
        p.compile(migraphx::ref::target{});
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<int> results_vector(6 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(migraphx::verify_range(result.get_shape().lens(), std::vector<std::size_t>({6, 2})));
        EXPECT(
            migraphx::verify_range(result.get_shape().strides(), std::vector<std::size_t>({2, 1})));
    }

    {
        migraphx::program p;
        auto* mm               = p.get_main_module();
        int axis               = -2;
        std::vector<int> data0 = {0, 1, 2, 3};
        std::vector<int> data1 = {4, 5, 6, 7, 8, 9};
        std::vector<int> data2 = {10, 11};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::int32_type, {1, 2}};
        auto l0 = mm->add_literal(migraphx::literal{s0, data0});
        auto l1 = mm->add_literal(migraphx::literal{s1, data1});
        auto l2 = mm->add_literal(migraphx::literal{s2, data2});
        mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
        p.compile(migraphx::ref::target{});
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<int> results_vector(6 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(migraphx::verify_range(result.get_shape().lens(), std::vector<std::size_t>({6, 2})));
        EXPECT(
            migraphx::verify_range(result.get_shape().strides(), std::vector<std::size_t>({2, 1})));
    }
}

TEST_CASE(contiguous_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(migraphx::literal{a_shape, data});
    mm->add_instruction(migraphx::make_op("contiguous"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<size_t> new_lens    = {1, 3, 2, 2};
    std::vector<size_t> new_strides = {12, 1, 6, 3};
    EXPECT(migraphx::verify_range(results_vector, data));
}

TEST_CASE(conv2d_padding_stride_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.14601797, -0.13000923, 0.06521662,  0.06178288,  -0.11083675, 0.10154136,  0.09990512,
        0.06030385,  -0.11374587, -0.17523311, -0.14344215, 0.17802463,  0.06300922,  -0.15325832,
        0.07066704,  0.05166031,  0.00615084,  -0.02606523, 0.08083995,  -0.17913306, 0.0624622,
        0.0735731,   -0.04198661, -0.0164391,  -0.06374192, 0.16569914,  0.10681538,  0.07370754,
        0.02802075,  0.00282027,  0.15104802,  -0.11084409, -0.00197773, 0.07924436,  0.03528272,
        0.04765259,  -0.15896152, 0.07917164,  0.12125669,  -0.1154705,  -0.11999125, 0.12749968,
        -0.06269585, 0.18658121,  -0.03944227, 0.0111798,   -0.17731084, 0.11789055,  -0.09982193,
        0.08142821,  0.0729029,   0.11303909,  0.12735154,  0.03885292};

    std::vector<float> s = {-0.20817225,
                            0.87965256,
                            0.14958936,
                            -1.24887264,
                            -0.06540672,
                            0.20778663,
                            0.40456355,
                            -0.99900877,
                            0.4917807,
                            0.1994698,
                            0.64205718,
                            0.37798831,
                            -0.25315839,
                            0.44276932,
                            -0.16138598,
                            0.79344082};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}), al, cl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(conv2d_padding_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.16115488, -0.09800646, -0.05412646, 0.10475694,  0.00555485,  -0.12667653, 0.0458357,
        -0.02656217, -0.16338061, 0.15037455,  0.0102711,   0.01303349,  0.05242859,  0.02034754,
        0.04751867,  -0.17038961, -0.1434752,  -0.10770349, 0.05676742,  -0.15838449, 0.10128359,
        -0.18958683, 0.11954515,  0.10758857,  -0.01058291, -0.12797487, 0.08971019,  0.18793164,
        -0.00881396, -0.06588994, -0.13321903, -0.03300409, 0.01439607,  0.07618178,  -0.11556662,
        0.00764295,  0.12956454,  -0.08937147, -0.12763587, 0.04674943,  0.05765297,  0.11336918,
        0.14747436,  -0.06199479, -0.01166052, -0.12432006, -0.04494537, -0.17581205, 0.09475745,
        0.1149437,   -0.1014564,  0.0274073,   -0.01323579, -0.11092556};

    std::vector<float> s = {
        -0.0201216,  0.40407312,  -0.39005592, -0.0631946,  0.37963012,  -0.64611685, 0.1349397,
        -0.54113752, 0.28533003,  0.27667275,  -0.16442731, -0.181494,   0.30564839,  0.58744538,
        0.32015014,  0.24969585,  -0.27367792, -0.53308117, 0.41236052,  0.26136363,  -0.01489828,
        0.57652152,  -0.38506854, 0.119615,    0.0437076,   0.04779706,  0.57887721,  0.23126155,
        0.05695833,  -0.68200272, 0.02063358,  -0.10267162, 0.8062973,   -0.38149622, -0.40134856,
        -0.03353126, 0.38991132,  -0.3478111,  0.03661491,  0.25783631,  0.62772679,  -0.1961118,
        0.76423508,  -0.36241418, -0.20994355, -0.12368261, -0.9406727,  0.02340185,  -0.08793129,
        -0.02471633, -0.58163726, -0.02211772, -0.42014724, 0.77525634,  0.504951,    -0.20537445,
        -0.20369984, -0.83037728, -1.40423918, -0.46160448, -0.22944322, 0.36074194,  0.49579027,
        0.46527559};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {1, 1}}}), al, cl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector(64);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(conv2d_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        2.82721668e-02,  6.44195229e-02,  1.53499246e-02,  1.72468081e-01,  -6.33238107e-02,
        9.49496776e-02,  1.40258059e-01,  -7.92879611e-02, -1.29301161e-01, 3.11307609e-03,
        -1.90624535e-01, 1.13238767e-01,  -2.80647576e-02, 3.12882811e-02,  -3.52091640e-02,
        3.33581865e-02,  6.43158704e-02,  7.40238279e-02,  -1.00106120e-01, -9.56912562e-02,
        1.44342467e-01,  9.40258950e-02,  6.36333972e-02,  1.66158378e-03,  -8.91554281e-02,
        2.58734226e-02,  1.70919895e-02,  1.78214177e-01,  8.84564668e-02,  8.98126513e-02,
        -1.63809001e-01, 1.37802169e-01,  1.66439757e-01,  -1.45631135e-02, 1.88469887e-04,
        4.76950556e-02,  -1.91969007e-01, -1.76233292e-01, -7.70473927e-02, 1.14828631e-01,
        1.76608220e-01,  -1.50728196e-01, 1.99946314e-02,  -5.88052124e-02, 1.31612435e-01,
        1.61106288e-02,  -1.35080189e-01, 1.49512306e-01,  3.86456847e-02,  1.29330024e-01,
        -3.22975963e-02, -5.60784787e-02, -5.41997552e-02, 4.78562862e-02};

    std::vector<float> s = {0.27039781,
                            0.19105849,
                            -0.06339942,
                            -0.65087199,
                            0.40867025,
                            0.05063812,
                            -0.14907975,
                            0.49018705,
                            -0.49197209,
                            0.33236548,
                            -0.39374301,
                            0.16012701,
                            0.06574871,
                            0.71606487,
                            -0.55201721,
                            -0.46427044};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(migraphx::make_op("convolution"), al, cl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(conv3d_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        2.82721668e-02,  6.44195229e-02,  1.53499246e-02,  1.72468081e-01,  -6.33238107e-02,
        9.49496776e-02,  1.40258059e-01,  -7.92879611e-02, -1.29301161e-01, 3.11307609e-03,
        -1.90624535e-01, 1.13238767e-01,  -2.80647576e-02, 3.12882811e-02,  -3.52091640e-02,
        3.33581865e-02,  6.43158704e-02,  7.40238279e-02,  -1.00106120e-01, -9.56912562e-02,
        1.44342467e-01,  9.40258950e-02,  6.36333972e-02,  1.66158378e-03,  -8.91554281e-02,
        2.58734226e-02,  1.70919895e-02,  1.78214177e-01,  8.84564668e-02,  8.98126513e-02,
        -1.63809001e-01, 1.37802169e-01,  1.66439757e-01,  -1.45631135e-02, 1.88469887e-04,
        4.76950556e-02,  -1.91969007e-01, -1.76233292e-01, -7.70473927e-02, 1.14828631e-01,
        1.76608220e-01,  -1.50728196e-01, 1.99946314e-02,  -5.88052124e-02, 1.31612435e-01,
        1.61106288e-02,  -1.35080189e-01, 1.49512306e-01,  3.86456847e-02,  1.29330024e-01,
        -3.22975963e-02, -5.60784787e-02, -5.41997552e-02, 4.78562862e-02};

    std::vector<float> s = {0.27039781,
                            0.19105849,
                            -0.06339942,
                            -0.65087199,
                            0.40867025,
                            0.05063812,
                            -0.14907975,
                            0.49018705,
                            -0.49197209,
                            0.33236548,
                            -0.39374301,
                            0.16012701,
                            0.06574871,
                            0.71606487,
                            -0.55201721,
                            -0.46427044};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4, 1}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3, 1}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        al,
        cl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(cos_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-1, 0, 1};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("cos"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return cosf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(cosh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data = {-1.0, 2.0, -3.0, 4.0};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("cosh"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return coshf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(deconv_1d_test)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3}};
    std::vector<float> x_data{0, 0.5, 1};
    std::vector<float> w_data{0.5, 0.5, 0.5};

    std::vector<float> gold{0, 0.25, 0.75, 0.75, 0.5};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(
        migraphx::make_op("deconvolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        x,
        w);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(deconv_3d_test)
{
    migraphx::shape s_1{migraphx::shape::float_type, {1, 1, 1, 2, 3}};
    migraphx::shape s_2{migraphx::shape::float_type, {1, 1, 3, 2, 3}};
    std::vector<float> x_data{0.8471, -0.4195, -2.2749, 1.2491, 0.1722, 0.3246};
    std::vector<float> w_data{0.6478,
                              -0.1985,
                              0.0633,
                              -0.3479,
                              2.7056,
                              -0.1440,
                              -1.1229,
                              -0.7507,
                              -1.3151,
                              0.8884,
                              -0.1859,
                              -0.3407,
                              -1.1544,
                              -1.5893,
                              1.6265,
                              -1.4624,
                              0.3812,
                              -1.5378};

    std::vector<float> gold{0.5488,  -0.4399, -1.3369, 0.4251,  -0.1439, 0.5145,  2.3015,  -0.2104,
                            -6.1482, 0.3482,  -0.4346, 3.3197,  0.1731,  0.8533,  -0.0467, -0.9512,
                            -0.1649, 1.7553,  2.2594,  2.9917,  -0.6500, -1.6612, -4.3680, 0.0957,
                            0.3482,  1.1097,  -0.0792, -0.1692, -0.1190, -0.1106, -0.9779, -0.8621,
                            4.6707,  2.9332,  -3.7001, -2.6808, -1.2476, 3.2475,  -0.4578, 4.0263,
                            -1.8267, 0.2243,  -2.3299, -0.1411, -0.4991};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s_1, x_data});
    auto w   = mm->add_literal(migraphx::literal{s_2, w_data});

    mm->add_instruction(
        migraphx::make_op("deconvolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        x,
        w);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(deconv_test)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::vector<float> gold{0,  1,  3, 3,  2,  3,  8,  15, 12, 7,  9,  21, 36,
                            27, 15, 9, 20, 33, 24, 13, 6,  13, 21, 15, 8};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(migraphx::make_op("deconvolution"), x, w);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(dequantizelinear)
{
    { /*uint8*/
        migraphx::shape xs{migraphx::shape::uint8_type, {1, 3, 3}};
        std::vector<uint8_t> xv = {0, 1, 2, 5, 10, 50, 100, 150, 250};
        migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
        std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
        migraphx::shape zs{migraphx::shape::uint8_type, {1, 3, 3}};
        std::vector<uint8_t> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto create_program     = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            auto z   = mm->add_literal(zs, zv);
            mm->add_instruction(migraphx::make_op("dequantizelinear"), x, s, z);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::ref::target{});
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(9);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0, 2, 4, 10, 20, 100, 200, 300, 500};
        EXPECT(results_vector == gold);
    }

    { /*int8*/
        migraphx::shape xs{migraphx::shape::int8_type, {1, 3, 3}};
        std::vector<int8_t> xv = {-128, -100, -50, -1, 0, 1, 50, 100, 127};
        migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
        std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
        auto create_program   = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            mm->add_instruction(migraphx::make_op("dequantizelinear"), x, s);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::ref::target{});
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(9);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{-256, -200, -100, -2, 0, 2, 100, 200, 254};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(div_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data1 = {-1.0f, 0.5f, 1.0f};
    std::vector<float> data2 = {1.0f, 2.0f, 4.0f};
    auto l1                  = mm->add_literal(migraphx::literal{s, data1});
    auto l2                  = mm->add_literal(migraphx::literal{s, data2});
    mm->add_instruction(migraphx::make_op("div"), l1, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(data1.size());
    std::transform(data1.begin(), data1.end(), data2.begin(), gold.begin(), std::divides<float>());
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(elu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l      = mm->add_literal(migraphx::literal{s, {-1.0, 2.0, -3.0, 4.0}});
    float alpha = 0.5;
    mm->add_instruction(migraphx::make_op("elu", {{"alpha", alpha}}), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{elu(alpha, -1), elu(alpha, 2), elu(alpha, -3), elu(alpha, 4)};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(equal_brcst_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {3, 3}};
    auto l0 =
        mm->add_literal(migraphx::literal{s0, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    migraphx::shape s1{migraphx::shape::float_type, {3, 1}};
    auto l1  = mm->add_literal(migraphx::literal{s1, {1.1, -1.5, 0.0}});
    auto bl1 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), l1);
    auto eq  = mm->add_instruction(migraphx::make_op("equal"), l0, bl1);
    auto r   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        eq);
    mm->add_return({r});

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {true, false, false, false, true, false, true, false, false};
    EXPECT(results_vector == gold);
}

TEST_CASE(equal_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    auto l0 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    auto l1 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1}});
    auto eq = mm->add_instruction(migraphx::make_op("equal"), l0, l1);
    auto r  = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        eq);
    mm->add_return({r});

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {true, false, false, false, true, false, true, false, false};
    EXPECT(results_vector == gold);
}

TEST_CASE(erf_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4}};
    std::vector<float> data = {0.73785057, 1.58165966, -0.43597795, -0.01677432};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("erf"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return erff(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(exp_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data{-1, 0, 1};
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("exp"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return expf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(floor_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    std::vector<float> data = {1.1, 1.5, 0.6, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("floor"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return floor(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(fp16_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {1}};
    migraphx::half a{1.5};
    migraphx::half b{2.5};
    migraphx::half c{4.0};
    auto l0 = mm->add_literal(migraphx::literal{s, {a}});
    auto l1 = mm->add_literal(migraphx::literal{s, {b}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<migraphx::half> results_vector(1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<migraphx::half> gold{c};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(fp32_fp16_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = mm->add_literal(migraphx::literal(s, data));
        auto l2 = mm->add_literal(migraphx::literal(s, data));
        mm->add_instruction(migraphx::make_op("add"), l1, l2);
        return p;
    };

    auto test_case = [&](std::vector<std::string>&& op_names) {
        std::vector<float> gold_res = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
        auto p                      = create_program();
        migraphx::quantize_fp16(p, op_names);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> res;
        result.visit([&](auto output) { res.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res, gold_res));
    };

    test_case({"all"});
    test_case({"add"});
}

TEST_CASE(gather_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{0, 2};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = 0;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{-3, -1};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = 0;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{0, 2};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = 1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{0, 2};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        // scalar index
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{0};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> golden = {0.5f, 3.5f, 6.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        // scalar index
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{-3};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> golden = {0.5f, 3.5f, 6.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        // scalar index
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{0};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> golden = {0.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(res_data, golden));
    }
}

TEST_CASE(globalavgpool_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op    = migraphx::op::pooling{"average"};
    auto lens  = s.lens();
    op.lengths = {lens[2], lens[3]};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.575, 0.375};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(globalmaxpool_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op    = migraphx::op::pooling{"max"};
    auto lens  = s.lens();
    op.lengths = {lens[2], lens[3]};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.9, 0.7};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(greater_brcst_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {3, 3}};
    auto l0 =
        mm->add_literal(migraphx::literal{s0, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    migraphx::shape s1{migraphx::shape::float_type, {3, 1}};
    auto l1  = mm->add_literal(migraphx::literal{s1, {1.1, -1.5, 0.0}});
    auto bl1 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), l1);
    auto gr  = mm->add_instruction(migraphx::make_op("greater"), l0, bl1);
    auto r   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        gr);
    mm->add_return({r});

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, true, false, true, false, true, false, true, false};
    EXPECT(results_vector == gold);
}

TEST_CASE(greater_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    auto l0 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    auto l1 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1}});
    auto gr = mm->add_instruction(migraphx::make_op("greater"), l0, l1);
    auto r  = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        gr);
    mm->add_return({r});

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, false, true, true, false, true, false, false, true};
    EXPECT(results_vector == gold);
}

TEST_CASE(identity_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<int> data{1, 2, 3, 4};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("identity"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::equal(data.begin(), data.end(), results_vector.begin()));
}

TEST_CASE(if_literal_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", cond_s);

        migraphx::shape s{migraphx::shape::float_type, {5}};

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {1, 2, 3, 4, 5};
        auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
        then_mod->add_return({l1});

        auto* else_mod           = p.create_module("If_0_else");
        std::vector<float> data2 = {5, 4, 3, 2, 1};
        auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
        else_mod->add_return({l2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({r});

        return p;
    };

    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::ref::target());
        std::vector<char> c_data = {static_cast<char>(cond)};
        migraphx::shape cs{migraphx::shape::bool_type};
        migraphx::parameter_map m;
        m["cond"] = migraphx::argument(cs, c_data.data());

        auto res = p.eval(m).back();
        std::vector<float> ret;
        res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });

        return ret;
    };

    // then branch
    {
        std::vector<float> gold_ret = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto ret                    = run_prog(true);
        EXPECT(gold_ret == ret);
    }

    // else branch
    {
        std::vector<float> gold_ret = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        auto ret                    = run_prog(false);
        EXPECT(gold_ret == ret);
    }
}

TEST_CASE(if_param_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", cond_s);
        migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
        auto x                   = mm->add_parameter("x", ds);
        auto y                   = mm->add_parameter("y", ds);
        std::vector<float> data2 = {-0.258047, 0.360394, 0.536804, -0.577762, 1.0217, 1.02442};
        auto l2                  = mm->add_literal(migraphx::literal(ds, data2));
        auto sum                 = mm->add_instruction(migraphx::make_op("add"), x, l2);

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {0.384804, -1.77948, -0.453775, 0.477438, -1.06333, -1.12893};
        auto l1                  = then_mod->add_literal(migraphx::literal(ds, data1));
        auto tx                  = then_mod->add_parameter("x", ds);
        auto a1                  = then_mod->add_instruction(migraphx::make_op("add"), tx, l1);
        then_mod->add_return({a1});

        auto* else_mod = p.create_module("If_0_else");
        auto ey        = else_mod->add_parameter("y", ds);
        auto a2        = else_mod->add_instruction(migraphx::make_op("mul"), ey, sum);
        else_mod->add_return({a2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond, x, y}, {then_mod, else_mod});
        auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({r});

        return p;
    };

    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::ref::target());
        std::vector<char> c_data = {static_cast<char>(cond)};
        migraphx::shape cs{migraphx::shape::bool_type};
        migraphx::parameter_map m;
        m["cond"] = migraphx::argument(cs, c_data.data());
        migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data_x(ds.elements(), 1);
        m["x"] = migraphx::argument(ds, data_x.data());
        std::vector<float> data_y(ds.elements(), 2);
        m["y"] = migraphx::argument(ds, data_y.data());

        auto res = p.eval(m).back();
        std::vector<float> ret;
        res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });
        std::cout << std::endl;

        return ret;
    };

    // then branch
    {
        std::vector<float> gold_ret = {
            1.384804, -0.77947998, 0.54622501, 1.477438, -0.063330054, -0.12892997};
        auto ret = run_prog(true);
        EXPECT(gold_ret == ret);
    }

    // else branch
    {
        std::vector<float> gold_ret = {
            1.483906, 2.720788, 3.0736079, 0.84447598, 4.0433998, 4.04884};
        auto ret = run_prog(false);
        EXPECT(gold_ret == ret);
    }
}

TEST_CASE(if_pl_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        migraphx::shape s{migraphx::shape::float_type, {5}};
        auto cond = mm->add_parameter("cond", cond_s);
        auto x    = mm->add_parameter("x", s);

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {1, 2, 3, 4, 5};
        auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
        then_mod->add_return({l1, x});

        auto* else_mod           = p.create_module("If_0_else");
        std::vector<float> data2 = {5, 4, 3, 2, 1};
        auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
        auto s2                  = else_mod->add_instruction(migraphx::make_op("add"), x, l2);
        else_mod->add_return({s2, l2});

        auto ret     = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto outline = mm->add_outline(s);
        auto r = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({outline, r});

        return p;
    };

    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::ref::target());
        std::vector<char> c_data = {static_cast<char>(cond)};
        migraphx::shape cs{migraphx::shape::bool_type};
        migraphx::parameter_map m;
        m["cond"] = migraphx::argument(cs, c_data.data());
        migraphx::shape ds{migraphx::shape::float_type, {5}};
        std::vector<float> data(ds.elements(), 1);
        m["x"] = migraphx::argument(ds, data.data());

        auto res = p.eval(m).back();
        std::vector<float> ret;
        res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });

        return ret;
    };

    // then branch
    {
        std::vector<float> gold_ret = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto ret                    = run_prog(true);
        EXPECT(gold_ret == ret);
    }

    // else branch
    {
        std::vector<float> gold_ret = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f};
        auto ret                    = run_prog(false);
        EXPECT(gold_ret == ret);
    }
}

TEST_CASE(im2col_3x3_no_pad_identity_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {3, 3};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, input));
}

TEST_CASE(im2col_3x3_no_pad_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {4, 4};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<int> correct = {0, 1, 2, 4, 5, 6,  8,  9,  10, 1, 2, 3, 5, 6,  7,  9,  10, 11,
                                4, 5, 6, 8, 9, 10, 12, 13, 14, 5, 6, 7, 9, 10, 11, 13, 14, 15};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, correct));
}

TEST_CASE(im2col_3x3_stride_2_no_pad_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {6, 6};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{2, 2};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<int> correct = {0,  1,  2,  6,  7,  8,  12, 13, 14, 2,  3,  4,
                                8,  9,  10, 14, 15, 16, 12, 13, 14, 18, 19, 20,
                                24, 25, 26, 14, 15, 16, 20, 21, 22, 26, 27, 28};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, correct));
}

TEST_CASE(im2col_3x3_with_channels_identity_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {3, 3};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 2;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, input));
}

TEST_CASE(im2col_3x3_with_padding_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {2, 2};
    std::vector<std::size_t> padding{1, 1};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<int> correct = {0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0,
                                0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, correct));
}

TEST_CASE(imagescaler_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto img           = mm->add_literal(migraphx::literal{s,
                                                 {0.2,
                                                  0.3,
                                                  0.5,
                                                  0.4,

                                                  0.7,
                                                  0.8,
                                                  0.1,
                                                  0.9,

                                                  0.15,
                                                  0.25,
                                                  0.35,
                                                  0.45}});
    auto scale_val     = mm->add_literal(2.f);
    auto scaled_tensor = mm->add_instruction(
        migraphx::make_op("scalar", {{"scalar_bcst_dims", s.lens()}}), scale_val);
    auto img_scaled = mm->add_instruction(migraphx::make_op("mul"), img, scaled_tensor);
    auto bias_vals  = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0.01, 0.02, 0.03}});
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s.lens()}}), bias_vals);
    mm->add_instruction(migraphx::make_op("add"), img_scaled, bias_bcast);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.41,
                               0.61,
                               1.01,
                               0.81,

                               1.42,
                               1.62,
                               0.22,
                               1.82,

                               0.33,
                               0.53,
                               0.73,
                               0.93};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(leaky_relu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = mm->add_literal(migraphx::literal{s, {-1.f, 0.f, 1.f}});
    mm->add_instruction(migraphx::make_op("leaky_relu", {{"alpha", 0.01}}), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.01f, 0.f, 1.f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(less_brcst_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {3, 3}};
    auto l0 =
        mm->add_literal(migraphx::literal{s0, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    migraphx::shape s1{migraphx::shape::float_type, {3, 1}};
    auto l1  = mm->add_literal(migraphx::literal{s1, {1.1, -1.5, 0.0}});
    auto bl1 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), l1);
    auto le  = mm->add_instruction(migraphx::make_op("less"), l0, bl1);
    auto r   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        le);
    mm->add_return({r});

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, false, true, false, false, false, false, false, true};
    EXPECT(results_vector == gold);
}

TEST_CASE(less_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    std::vector<float> data1 = {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0};
    std::vector<float> data2 = {1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1};
    auto l0                  = mm->add_literal(migraphx::literal{s, data1});
    auto l1                  = mm->add_literal(migraphx::literal{s, data2});
    auto le                  = mm->add_instruction(migraphx::make_op("less"), l0, l1);
    auto r                   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        le);
    mm->add_return({r});

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold(data1.size());
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](float n1, float n2) -> bool {
            return n1 < n2;
        });
    EXPECT(results_vector == gold);
}

TEST_CASE(log_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data = {1, 2, 3};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("log"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return logf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(logical_and_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::bool_type, {4}};
    std::vector<bool> data1{true, false, true, false};
    std::vector<bool> data2{true, true, false, false};
    auto l1 = mm->add_literal(migraphx::literal{s, data1});
    auto l2 = mm->add_literal(migraphx::literal{s, data2});
    mm->add_instruction(migraphx::make_op("logical_and"), l1, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold(data2.size());
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](bool n1, bool n2) -> bool {
            return n1 and n2;
        });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(logical_or_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::bool_type, {4}};
    std::vector<bool> data1{true, false, true, false};
    std::vector<bool> data2{true, true, false, false};
    auto l1 = mm->add_literal(migraphx::literal{s, data1});
    auto l2 = mm->add_literal(migraphx::literal{s, data2});
    mm->add_instruction(migraphx::make_op("logical_or"), l1, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold(data1.size());
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](bool n1, bool n2) -> bool {
            return n1 or n2;
        });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(logical_xor_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::bool_type, {4}};
    std::vector<bool> data1{true, false, true, false};
    std::vector<bool> data2{true, true, false, false};
    auto l1 = mm->add_literal(migraphx::literal{s, data1});
    auto l2 = mm->add_literal(migraphx::literal{s, data2});
    mm->add_instruction(migraphx::make_op("logical_xor"), l1, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, true, true, false};
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](bool n1, bool n2) -> bool {
            return n1 ^ n2;
        });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(logsoftmax_test_axis_0)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.135261, -2.843968, -0.659995, -0.488413, -1.051857, -2.812936, -0.250956, -0.353985,
        -1.155980, -0.603651, -0.211969, -0.175371, -1.336552, -3.885010, -1.871544, -0.837083,
        -0.887745, -0.433338, -1.158864, -4.911197, -1.147972, -0.666711, -0.996874, -0.981418,
        -0.851145, -0.853988, -0.858112, -2.067420, -0.059956, -0.727436, -0.950881, -0.429689,
        -0.061906, -1.505332, -1.210277, -0.377970, -0.791448, -1.655428, -1.827253, -0.304828,
        -0.020762, -0.167101, -0.567346, -0.530319, -1.045094, -0.376648, -0.007391, -0.381670,
        -0.720302, -0.460499, -0.469651, -0.556740, -0.554628, -0.551582};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 0;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_1)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.550468, -2.132973, -1.549746, -0.650533, -1.051529, -2.248570, -0.141017, -2.028357,
        -1.947730, -1.511324, -0.166597, -0.379726, -1.965689, -1.172109, -1.475721, -2.700831,
        -1.537011, -0.658754, -1.596017, -3.353137, -2.266743, -1.084197, -1.076214, -0.406712,
        -2.743019, -0.425526, -1.079083, -2.139486, -1.270584, -1.024088, -1.154231, -3.201762,
        -0.888957, -0.532855, -3.103583, -1.221339, -1.355980, -3.531678, -1.438510, -0.975194,
        -0.080261, -1.162697, -1.568557, -1.398519, -1.322129, -0.470660, -0.370953, -0.907343,
        -1.179017, -3.312239, -1.286363, -1.586076, -0.345100, -0.824173};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 1;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_2)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.495957, -1.031212, -0.245531, -2.013726, -1.339125, -2.465619, -1.356652, -0.964037,
        -2.019250, -0.214522, -0.289569, -0.234392, -2.086591, -2.684439, -2.851651, -2.674176,
        -1.697424, -1.889155, -0.401029, -3.064586, -1.173030, -1.306912, -2.177020, -0.834262,
        -2.818177, -0.174415, -1.361105, -1.024571, -0.106766, -1.167645, -1.072650, -2.576522,
        -0.569261, -1.207483, -3.679894, -2.095913, -0.504264, -3.039291, -1.290559, -1.156812,
        -0.126453, -0.551493, -2.506384, -2.646261, -1.905195, -0.206994, -0.191369, -0.959754,
        -1.948685, -3.671233, -0.875521, -3.111952, -1.905644, -1.6076011};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 2;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_3)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.336904, -3.475825, -1.366154, -0.279366, -2.208430, -2.010934, -0.225511, -2.436562,
        -2.167785, -1.572415, -1.784104, -0.470789, -1.067459, -1.801948, -0.711023, -2.307197,
        -1.467087, -0.400681, -0.426983, -3.740518, -1.127681, -1.078919, -2.599005, -0.534965,
        -2.561400, -0.567617, -1.033025, -2.097713, -0.520463, -1.262245, -1.763230, -2.607658,
        -0.281299, -0.814243, -2.627210, -0.724131, -0.655704, -2.123055, -1.018163, -2.480634,
        -0.382599, -1.451479, -1.843102, -0.915303, -0.818078, -1.316929, -0.508875, -2.033541,
        -1.487672, -2.417791, -0.378360, -2.568531, -0.569794, -1.028032};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 3;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(lrn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1, 5, 1, 1}};
    auto l = mm->add_literal(migraphx::literal{s, {-2.0f, 1.0f, 0.f, 1.0f, 2.0f}});
    mm->add_instruction(
        migraphx::make_op("lrn", {{"alpha", 0.0001}, {"beta", 0.75}, {"bias", 1}, {"size", 5}}), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(5);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2 / 1.000075, 1 / 1.00009, 0 / 1.000145, 1 / 1.00009, 2 / 1.000075};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(max_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l0       = mm->add_literal(migraphx::literal{s, {1, 4, 3}});
    auto l1       = mm->add_literal(migraphx::literal{s, {2, 8, 6}});
    auto l2       = mm->add_literal(migraphx::literal{s, {7, 5, 9}});
    auto curr_max = mm->add_instruction(migraphx::make_op("max"), l0, l1);
    mm->add_instruction(migraphx::make_op("max"), curr_max, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{7, 8, 9};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(maxpool_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        -2.1314404,  -1.63041711, 1.54562736,  1.04625261,  -1.42931843, -0.48703974, 0.4065806,
        -0.1524526,  1.30775225,  0.45538983,  -0.06631992, -1.75332725, 1.33493888,  0.47327688,
        0.36873096,  1.18358743,  -0.34640595, 1.22098756,  0.01946825,  -0.20238149, 0.43348005,
        -0.67991608, -0.83041084, 0.93537551,  0.70241445,  -0.5654031,  -1.30899191, -0.26735824,
        -0.52444768, 1.99097753,  1.86504853,  -0.26506025, 0.26236168,  0.43763575,  0.95300823,
        -1.02733946, -0.74655169, -0.5374338,  -0.28901565, -0.59789604, 0.5310151,   0.99125904,
        0.40609556,  -1.57175648, 0.22031412,  1.45862222,  0.53217483,  1.39087725,  1.00170159,
        -0.87175864, -1.7204628,  -1.72008383, -0.38656762, -0.01443311, 1.46645272,  -1.39995027,
        0.22505587,  -0.43461126, -0.05511411, -0.79950953, -0.01439556, 0.08795211,  1.18943918,
        -0.84079367, -1.73383629, -0.55662078, -0.30626822, -0.67339015, 0.44179603,  0.54316711,
        0.40899998,  -0.27831686, -1.11900508, -0.0881724,  0.35483059,  2.36277103,  -0.04765317,
        -0.36865309, 0.73814237,  1.47151589,  1.36546791,  -0.32649881, -1.0517807,  2.24768877,
        0.68883753,  0.58646208,  -0.91017133, -0.50462508, -0.4013325,  -0.72348958, -0.47368807,
        0.35285577,  -1.01817429, -0.5152272,  0.60321307,  0.43521205,  -0.23733577, 0.66427642,
        0.82949388,  0.82443929,  0.71550399,  0.34561086,  0.68570769,  -0.40718508, -1.20350206,
        0.15793853,  -2.31013632, -0.07934658, -0.09348056, 0.36576006,  2.46601582,  0.11090943,
        0.9144392,   0.56759721,  -0.22112127, -0.21955389, 0.72474903,  -1.28448462, 1.53285873,
        0.37437943,  0.31409341,  1.95433736,  0.91620457,  0.86205518,  1.24365854,  0.19248386,
        0.22526583,  0.13462132,  -0.27561715, -2.06446075, -0.02306402, -1.38278747, 1.1411345,
        1.31293464,  -1.86041689, 1.06763375,  -0.26541466, 1.4545635,   1.11430049,  -0.66491818,
        0.87101674,  0.67768967,  -1.02062869, -1.05031872, -2.2764678,  -2.0200038,  0.37592548,
        -0.26701379, -0.83388507, 0.19403623,  1.00968623,  0.11020003,  1.16736257,  -1.1160326,
        0.47346735,  0.6126079,   -0.19135755, 1.33624589,  -0.29802522, -0.57873946, -1.06555879,
        -0.20686582, 1.36892557,  -0.19937795, 0.8649236,   -1.40126073, 1.53441942,  0.34682792,
        -1.31724346, -1.32898355, 2.40126371,  0.07845283,  1.35732043,  -0.63678312, 0.39429256,
        -1.36487007, -0.31026676, -0.44981545, -0.28994772, -0.14657612, -1.75206447, -0.70612341,
        1.20071781,  -1.64647579, -0.7133292,  0.88494766,  0.52119428,  -2.77387547, 2.07681108,
        -0.90133125, 0.2847338,   0.6174528,   -0.20616426, -0.64263535, -1.08496261, 0.54275119,
        -0.88503587, 0.6629802,   1.47319221,  -1.05829155, -0.97027361, -0.93187737, -1.39954746,
        -0.52359426, -0.14743951, 1.51522756,  0.2078452,   -1.28156149, -1.19363916, -0.78680223,
        -0.89094824, 1.30212069,  -0.77974445, -0.58411664, 0.48764706,  -0.67132682};
    std::vector<float> c = {1.33493888, 1.54562736, 1.22098756, 1.33493888, 1.18358743, 1.99097753,
                            1.00170159, 1.45862222, 1.39087725, 1.46645272, 1.18943918, -0.01443311,
                            1.47151589, 2.36277103, 2.24768877, 0.68883753, 0.82949388, 0.71550399,
                            1.95433736, 2.46601582, 1.53285873, 1.95433736, 1.06763375, 1.4545635,
                            1.33624589, 1.16736257, 0.6126079,  1.36892557, 2.40126371, 1.53441942,
                            0.52119428, 2.07681108, 0.88494766, 1.51522756, 0.54275119, 0.6629802};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 6, 6}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    mm->add_instruction(
        migraphx::make_op(
            "pooling",
            {{"mode", "max"}, {"padding", {0, 0}}, {"stride", {2, 2}}, {"lengths", {3, 2}}}),
        al);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(36);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, c));
}

TEST_CASE(maxpool_test_1D_3D)
{
    // 1D case 1, input is 3D
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
        auto op    = migraphx::op::pooling{"max"};
        op.lengths = {2};
        op.padding = {0};
        op.stride  = {1};

        std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(op, l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();

        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0.3, 0.4, 0.4, 0.8, 0.9, 0.9, 0.7, 0.7, 0.6};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }

    // 1D case 2, input is 3D
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {2, 2, 5}};
        auto op    = migraphx::op::pooling{"max"};
        op.lengths = {2};
        op.padding = {0};
        op.stride  = {2};

        std::vector<float> data{0.4975, -0.1226, -0.0405, -0.2861, -0.1227, -0.6186, -0.9618,
                                0.6022, -0.1912, 1.1925,  0.5493,  0.1692,  -0.8039, -1.0281,
                                0.9907, 0.477,   1.5001,  -1.1603, -1.361,  1.2556};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(op, l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();

        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0.4975, -0.0405, -0.6186, 0.6022, 0.5493, -0.8039, 1.5001, -1.1603};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }

    // 1D case 2, input is 3D, ceil mode
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto s       = migraphx::shape{migraphx::shape::float_type, {2, 2, 5}};
        auto op      = migraphx::op::pooling{"max"};
        op.lengths   = {2};
        op.padding   = {0};
        op.stride    = {2};
        op.ceil_mode = true;

        std::vector<float> data{0.4975, -0.1226, -0.0405, -0.2861, -0.1227, -0.6186, -0.9618,
                                0.6022, -0.1912, 1.1925,  0.5493,  0.1692,  -0.8039, -1.0281,
                                0.9907, 0.477,   1.5001,  -1.1603, -1.361,  1.2556};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(op, l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();

        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0.4975,
                                -0.0405,
                                -0.1227,
                                -0.6186,
                                0.6022,
                                1.1925,
                                0.5493,
                                -0.8039,
                                0.9907,
                                1.5001,
                                -1.1603,
                                1.2556};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }

    // 3D, input is 5D
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 3, 3}};
        auto op    = migraphx::op::pooling{"max"};
        op.lengths = {2, 2, 2};
        op.padding = {0, 0, 0};
        op.stride  = {2, 2, 2};

        std::vector<float> data{
            -2.8029, 0.5861,  0.7015,  0.1297,  -1.44,   -1.9472, 0.7812,  2.408,   -0.3145,
            0.3405,  -0.9146, 0.0624,  1.5064,  -0.8345, 1.7977,  1.8949,  1.0073,  -0.2102,
            -0.042,  -0.7146, 0.6227,  -0.5263, -2.2598, 0.1713,  0.449,   0.5303,  -0.8622,
            -0.5691, 0.907,   -0.0569, -1.5348, -0.4109, -0.1461, -0.5445, 0.4266,  0.2282,
            1.3655,  -2.1519, 0.6068,  -0.2001, -0.4702, 0.3864,  1.7083,  0.9096,  0.4286,
            -1.8866, 0.7034,  0.0293,  1.4587,  0.7672,  -2.8614, 0.8124,  -0.053,  1.0449,
            0.845,   -0.0131, 0.1139,  -0.859,  -1.2681, -0.6337, -0.4644, 0.1938,  0.2889,
            0.9035,  0.7118,  -0.5767, 0.4577,  -0.0549, 0.2237,  0.5756,  0.0677,  -0.0223,
            -0.329,  0.2364,  2.7666,  -0.7417, -1.3196, -0.2655, 0.1698,  -0.1777, -0.9427,
            2.6859,  -0.7501, 0.5175,  1.0029,  -2.6436, -0.4388, -1.2348, -0.1539, -0.6229,
            -0.4136, 0.5085,  0.4136,  -0.6439, -1.1953, -0.406,  -0.0195, 0.1869,  -0.8664,
            1.1364,  0.5041,  0.0647,  0.1941,  -1.0819, -0.4629, -0.5107, 0.3612,  -0.3583};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(op, l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.5064, 1.3655, 0.9035, 2.6859};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
}

TEST_CASE(min_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l0       = mm->add_literal(migraphx::literal{s, {1, 4, 3}});
    auto l1       = mm->add_literal(migraphx::literal{s, {2, 8, 6}});
    auto l2       = mm->add_literal(migraphx::literal{s, {7, 5, 9}});
    auto curr_min = mm->add_instruction(migraphx::make_op("min"), l0, l1);
    mm->add_instruction(migraphx::make_op("min"), curr_min, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 4, 3};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(mul_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data1{-1, 0, 1};
    std::vector<float> data2{1, 2, 3};
    auto l1 = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    mm->add_instruction(migraphx::make_op("mul"), l1, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(data1.size());
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](float n1, float n2) -> float {
            return n1 * n2;
        });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(neg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1.0f, 1.3f, -1.2f, 0.0f, -100.f, 200.f};
    auto input              = mm->add_literal(migraphx::literal(s, data));
    auto ret                = mm->add_instruction(migraphx::make_op("neg"), input);
    mm->add_return({ret});
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(gold.begin(), gold.end(), gold.begin(), std::negate<float>());
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(not_test)
{
    // int32
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {4}};
        std::vector<float> data{0, 8, 1, -32};
        auto l1 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(migraphx::make_op("not"), l1);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<char> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<char> gold{1, 0, 0, 0};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }

    // bool
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::bool_type, {4}};
        std::vector<bool> data{false, false, true, true};
        auto l1 = mm->add_literal(migraphx::literal{s, {0, 0, 1, 1}});
        mm->add_instruction(migraphx::make_op("not"), l1);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<char> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<bool> gold(data.size());
        std::transform(data.begin(), data.end(), gold.begin(), [](bool n) -> bool { return !n; });
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
}

TEST_CASE(pad_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {1, 1, 1, 1}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(pad_test_highest_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op("pad",
                          {{"pads", {1, 1, 1, 1}}, {"value", std::numeric_limits<float>::max()}}),
        l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    const float x = std::numeric_limits<migraphx::half>::max();
    std::vector<float> gold{x, x, x, x, x, 1, 2, x, x, 3, 4, x, x, x, x, x};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(pad_test_lowest_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op(
            "pad", {{"pads", {1, 1, 1, 1}}, {"value", std::numeric_limits<float>::lowest()}}),
        l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    const float x = std::numeric_limits<migraphx::half>::lowest();
    std::vector<float> gold{x, x, x, x, x, 1, 2, x, x, 3, 4, x, x, x, x, x};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(pow_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data = {1, 2, 3};
    auto b                  = mm->add_literal(migraphx::literal{s, data});
    auto e                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("pow"), b, e);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return std::pow(n, n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(prefix_scan_sum_1d)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {6}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.0, 3.0, 6.0, 10.0, 15.0, 21.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_2d)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prefix_scan_sum_3d)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                2.0,
                                3.0,
                                1.0,
                                2.0,
                                3.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                2.0,
                                4.0,
                                6.0,
                                2.0,
                                4.0,
                                6.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                3.0,
                                6.0,
                                9.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                3.0,
                                6.0,
                                9.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 2}, {"exclusive", false}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prefix_scan_sum_exclusive)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {8}};
        auto input = migraphx::literal{s, {1, 2, 3, 4, 1, 2, 3, 4}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", true}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0.0, 1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 16.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", true}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0.0,
                                0.0,
                                0.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prefix_scan_sum_exclusive_reverse)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {6}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(
        migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", true}, {"reverse", true}}),
        l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{20.0, 18.0, 15.0, 11.0, 6.0, 0.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_negative_axis)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", -3}, {"exclusive", false}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                2.0,
                                3.0,
                                1.0,
                                2.0,
                                3.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                2.0,
                                4.0,
                                6.0,
                                2.0,
                                4.0,
                                6.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", -2}, {"exclusive", false}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                3.0,
                                6.0,
                                9.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                3.0,
                                6.0,
                                9.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", -1}, {"exclusive", false}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prefix_scan_sum_reverse)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {8}};
        auto input = migraphx::literal{s, {1, 2, 3, 4, 1, 2, 3, 4}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum",
                              {{"axis", 0}, {"exclusive", false}, {"reverse", true}}),
            l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{20.0, 19.0, 17.0, 14.0, 10.0, 9.0, 7.0, 4.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 2, 2}};
        auto input = migraphx::literal{s, {1, 2, 3, 4, 1, 2, 3, 4}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum",
                              {{"axis", 0}, {"exclusive", false}, {"reverse", true}}),
            l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{2.0, 4.0, 6.0, 8.0, 1.0, 2.0, 3.0, 4.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prelu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto x     = mm->add_literal(migraphx::literal{s, {-1, 0, 2}});
    auto slope = mm->add_literal(migraphx::literal{s, {2, 1, 2}});
    mm->add_instruction(migraphx::make_op("prelu"), x, slope);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2.0f, 0.0f, 2.0f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(quant_conv2d_padding_stride_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});
    mm->add_instruction(
        migraphx::make_op("quant_convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}), al, cl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<int32_t> s = {4521,
                              7014,
                              7830,
                              11952,
                              10515,
                              16734,
                              19737,
                              30906,
                              13161,
                              19542,
                              19494,
                              28800,
                              34707,
                              52590,
                              54729,
                              82746};
    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(quant_conv2d_padding_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});
    mm->add_instruction(
        migraphx::make_op("quant_convolution", {{"padding", {1, 1}}, {"stride", {1, 1}}}), al, cl);
    p.compile(migraphx::ref::target{});
    auto result            = p.eval({}).back();
    std::vector<int32_t> s = {
        4521,  6753,  7014,  4635,  6858,  10197, 10548, 6939,  7830,  11601, 11952, 7839,  5007,
        7383,  7590,  4953,  10515, 15987, 16734, 11277, 16821, 25506, 26586, 17874, 19737, 29826,
        30906, 20718, 13593, 20505, 21198, 14187, 13161, 19281, 19542, 12699, 18522, 27045, 27396,
        17739, 19494, 28449, 28800, 18639, 11919, 17319, 17526, 11289, 34707, 51843, 52590, 34893,
        51813, 77346, 78426, 52002, 54729, 81666, 82746, 54846, 36057, 53769, 54462, 36075};

    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(quant_conv2d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(migraphx::make_op("quant_convolution"), al, cl);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();

    std::vector<int32_t> s = {10197,
                              10548,
                              11601,
                              11952,
                              25506,
                              26586,
                              29826,
                              30906,
                              27045,
                              27396,
                              28449,
                              28800,
                              77346,
                              78426,
                              81666,
                              82746};

    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(quantizelinear)
{
    {
        migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> xv = {
            -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
        migraphx::shape ss{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> sv = {2, 2, 2, 4, 4, 4, 6, 6, 6, 2, 2, 2, 4, 4, 4, 6, 6, 6};
        migraphx::shape zs{migraphx::shape::int8_type, {2, 3, 3}};
        std::vector<uint8_t> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto create_program     = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            auto z   = mm->add_literal(zs, zv);
            mm->add_instruction(migraphx::make_op("quantizelinear"), x, s, z);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::ref::target{});
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(18);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{
            -128, 127, 65, -128, 1, 1, -1, 100, 92, -128, 127, 65, -128, 1, 1, -1, 100, 92};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> xv = {
            -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
        migraphx::shape ss{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
        auto create_program   = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            mm->add_instruction(migraphx::make_op("quantizelinear"), x, s);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::ref::target{});
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(18);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0, 255, 65, 0, 2, 2, 0, 255, 255, 0, 255, 65, 0, 2, 2, 0, 255, 255};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(recip_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{-0.5f, 0.1f, 0.5f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("recip"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2.0f, 10.0f, 2.0f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(reduce_max_axis0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{9, 10, 11, 12};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_max_axis01)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 1}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{11, 12};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_max_axis02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{10, 12};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{5.5, 7.5};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2, 3, 6, 7, 10, 11};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis12)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2.5f, 6.5f, 10.5f};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_int)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold{2, 6, 10};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_min_axis02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {0, 2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 3};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_min_axis1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {1}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 2, 5, 6, 9, 10};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_min_axis12)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 5, 9};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_prod_axis0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_prod", {{"axes", {0}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{6, 18, 12, 18};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{15, 18, 21, 24};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{33, 45};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{4, 6, 12, 14, 20, 22};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis12)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{10, 26, 42};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), l0);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{3, 7, 11, 15, 19, 23};
    EXPECT(results_vector == gold);
}

TEST_CASE(relu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = mm->add_literal(migraphx::literal{s, {-1.f, 0.f, 1.f}});
    mm->add_instruction(migraphx::make_op("relu"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.f, 0.f, 1.f};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(reshape_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), -3);
    {
        migraphx::program p;
        auto* mm                       = p.get_main_module();
        auto l                         = mm->add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> new_shape = {8, 3, 1, 1};
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), l);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector(3);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, data));
    }
    {
        migraphx::program p;
        auto* mm                       = p.get_main_module();
        auto l                         = mm->add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> new_shape = {1, 3, 4, 2};
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), l);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector(3);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, data));
    }
    {
        migraphx::program p;
        auto* mm                       = p.get_main_module();
        auto l                         = mm->add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> new_shape = {1, 3, 4, 2};
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), l);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector(3);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, data));
    }
}

TEST_CASE(reverse_test_axis0)
{
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1);
    migraphx::program p;
    auto* mm              = p.get_main_module();
    auto l                = mm->add_literal(migraphx::literal{in_shape, data});
    std::vector<int> axes = {0};
    mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> target_data = data;
    std::swap_ranges(target_data.begin(), target_data.begin() + 16, target_data.begin() + 16);
    EXPECT(migraphx::verify_range(results_vector, target_data));
}

TEST_CASE(reverse_test_axis1)
{
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1);
    migraphx::program p;
    auto* mm              = p.get_main_module();
    auto l                = mm->add_literal(migraphx::literal{in_shape, data});
    std::vector<int> axes = {1};
    mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> target_data = data;
    std::reverse(target_data.begin(), target_data.begin() + 16);
    std::reverse(target_data.end() - 16, target_data.end());
    EXPECT(migraphx::verify_range(results_vector, target_data));
}

TEST_CASE(reverse_test_axis10)
{
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1);
    migraphx::program p;
    auto* mm              = p.get_main_module();
    auto l                = mm->add_literal(migraphx::literal{in_shape, data});
    std::vector<int> axes = {1, 0};
    mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> target_data = data;
    std::reverse(target_data.begin(), target_data.begin() + 16);
    std::reverse(target_data.end() - 16, target_data.end());
    std::swap_ranges(target_data.begin(), target_data.begin() + 16, target_data.begin() + 16);
    EXPECT(migraphx::verify_range(results_vector, target_data));
}

TEST_CASE(round_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    auto l =
        mm->add_literal(migraphx::literal{s, {1.1, 1.5, 1.6, -1.1, -1.5, -1.6, 0.0, 2.0, -2.0}});
    mm->add_instruction(migraphx::make_op("round"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.0, 2.0, 2.0, -1.0, -2.0, -2.0, 0.0, 2.0, -2.0};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(rsqrt_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = mm->add_literal(migraphx::literal{s, {4.0, 16.0, 64.0}});
    mm->add_instruction(migraphx::make_op("rsqrt"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.5, 0.25, 0.125};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(scatter_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
        std::vector<float> vd(sd.elements(), 0.0f);

        migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
        std::vector<int> vi = {1, 0, 2, 0, 2, 1};

        migraphx::shape su{migraphx::shape::float_type, {2, 3}};
        std::vector<float> vu = {1.0, 1.1, 1.2, 2.0, 2.1, 2.2};

        auto ld = mm->add_literal(migraphx::literal{sd, vd});
        auto li = mm->add_literal(migraphx::literal{si, vi});
        auto lu = mm->add_literal(migraphx::literal{su, vu});
        auto r  = mm->add_instruction(migraphx::make_op("scatter", {{"axis", 0}}), ld, li, lu);
        mm->add_return({r});
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
        std::vector<float> vd(sd.elements(), 0.0f);

        migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
        std::vector<int> vi = {1, 0, -1, 0, 2, -2};

        migraphx::shape su{migraphx::shape::float_type, {2, 3}};
        std::vector<float> vu = {1.0, 1.1, 1.2, 2.0, 2.1, 2.2};

        auto ld = mm->add_literal(migraphx::literal{sd, vd});
        auto li = mm->add_literal(migraphx::literal{si, vi});
        auto lu = mm->add_literal(migraphx::literal{su, vu});
        auto r  = mm->add_instruction(migraphx::make_op("scatter", {{"axis", -2}}), ld, li, lu);
        mm->add_return({r});
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
        std::vector<float> vd(sd.elements(), 0.0f);

        migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
        std::vector<int> vi = {1, 0, 2, 0, 2, 1};

        migraphx::shape su{migraphx::shape::float_type, {2, 3}};
        std::vector<float> vu = {1.0, 1.1, 1.2, 2.0, 2.1, 2.2};

        auto ld = mm->add_literal(migraphx::literal{sd, vd});
        auto li = mm->add_literal(migraphx::literal{si, vi});
        auto lu = mm->add_literal(migraphx::literal{su, vu});
        auto r  = mm->add_instruction(migraphx::make_op("scatter", {{"axis", 1}}), ld, li, lu);
        mm->add_return({r});
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {1.1, 1.0, 1.2, 2.0, 2.2, 2.1, 0.0, 0.0, 0.0};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
}

TEST_CASE(sigmoid_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l = mm->add_literal(migraphx::literal{s, {-1, 2, -3, 4}});
    mm->add_instruction(migraphx::make_op("sigmoid"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{sigmoid(-1), sigmoid(2), sigmoid(-3), sigmoid(4)};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(sign_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto l = mm->add_literal(
        migraphx::literal{s, {1.02481645, 0.85643062, -0.03404123, -0.92791926, 0.0}});
    mm->add_instruction(migraphx::make_op("sign"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.0, 1.0, -1.0, -1.0, 0.0};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(sin_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data = {-1, 0, 1};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("sin"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return sinf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(sinh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data{-1.0, 2.0, -3.0, 4.0};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("sinh"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return sinhf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(slice_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<int> data(2 * 2 * 3);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}), l0);
        migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
        EXPECT(p.get_output_shapes().back() == s2);
        p.compile(migraphx::ref::target{});
        migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
        auto result           = p.eval({}).back();
        std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
        std::vector<int> results_vector(2 * 2 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(result.get_shape() == sresult);
    }
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<int> data(2 * 2 * 3);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {0, 1, 2}}, {"starts", {0, 0, 0}}, {"ends", {2, 2, 2}}}),
            l0);
        migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
        EXPECT(p.get_output_shapes().back() == s2);
        p.compile(migraphx::ref::target{});
        migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 3, 4, 6, 7, 9, 10};
        std::vector<int> results_vector(2 * 2 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify_range(results_vector, gold));
        EXPECT(result.get_shape() == sresult);
    }
}

TEST_CASE(softmax_simple_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {0.25, 0.75};
    std::vector<float> s = {0.377541, 0.622459};
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), al);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(softmax_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        -5.61869681e-01, 9.07827199e-01,  1.29255986e+00,  3.18533443e-02,  -1.22183852e-03,
        -2.83830553e-01, -1.03245842e+00, -9.28322077e-01, -8.82696748e-01, 1.11327164e-01,
        -9.20038462e-01, 8.47388089e-01,  2.51734018e-01,  1.50563884e+00,  2.23056650e+00,
        -6.17576987e-02, -1.00264274e-01, -6.10369384e-01, 1.17537189e+00,  -2.51560897e-01,
        -8.50333512e-01, -8.03578615e-01, -6.51194930e-01, -2.58137047e-01, 4.65528190e-01,
        3.23284641e-02,  -1.54700470e+00, 1.38096774e+00,  5.39869189e-01,  -7.56884992e-01,
        1.81503093e+00,  -2.11269641e+00, 1.92466557e+00,  1.77230799e+00,  2.21660900e+00,
        1.56777036e+00,  -2.08995026e-03, 3.50566894e-01,  -1.15042710e+00, -1.18577778e+00,
        8.90633047e-01,  -6.63949102e-02, 1.44661188e+00,  1.59215283e+00,  -2.56262213e-01,
        9.39079225e-01,  4.07298543e-02,  3.86590779e-01,  6.09607756e-01,  8.22331488e-01,
        -2.82126725e-01, -9.49052632e-01, -4.24012303e-01, -5.32990396e-01, -3.18386006e+00,
        3.27092171e-01,  -1.33315325e+00, 3.62459183e-01,  3.74710828e-01,  -1.30302286e+00,
        1.79680198e-01,  -4.51832324e-01, 4.34282750e-01,  -7.09520102e-01, 6.20333970e-01,
        -1.28712380e+00, 2.04130828e-01,  -7.70607769e-01, 1.61889160e+00,  -1.50951004e+00,
        -4.10505563e-01, -3.56566496e-02, -1.29747534e+00, -1.49967879e-01, 7.77626812e-01,
        -8.28408226e-02, 2.73412596e-02,  5.79780899e-03,  9.87900198e-02,  -7.95276761e-01,
        -1.38536084e+00, -6.63573861e-01, 3.89783204e-01,  -1.30670881e+00, -7.62425125e-01,
        -4.04883057e-01, 6.24344349e-01,  3.68128955e-01,  -1.01577950e+00, -3.06715906e-01,
        5.67961395e-01,  2.98198581e-01,  -1.63613629e+00, -3.75131965e-01, -6.75393403e-01,
        2.59172034e+00,  6.75538957e-01,  9.07939598e-02,  1.92257717e-01,  -1.21592450e+00,
        -2.73682117e-01, 1.25232983e+00,  -1.39969170e+00, -1.91483587e-01, 2.57732719e-01,
        3.10056299e-01,  1.41833842e+00,  -1.81386679e-01, 3.92868072e-01,  -8.14771175e-01,
        2.02392387e+00,  -9.42091495e-02, -3.77683818e-01, 2.05638766e+00,  2.93796062e-01,
        -6.02131486e-01, 2.70461679e-01,  -8.92358482e-01, 1.04388881e+00,  2.66154885e-01};

    std::vector<float> s = {
        0.30191708, 0.59879845, 0.50029165, 0.24915339, 0.36823985, 0.13190967, 0.0349741,
        0.18750034, 0.21905553, 0.27000085, 0.0547399,  0.56318235, 0.47422904, 0.78964758,
        0.91381913, 0.44601166, 0.47902739, 0.13120073, 0.4449684,  0.18766427, 0.15753111,
        0.07844277, 0.05120674, 0.36648798, 0.14637007, 0.13152322, 0.01560997, 0.29065287,
        0.49196178, 0.10550152, 0.81890774, 0.06369215, 0.62972021, 0.74931765, 0.67285055,
        0.35034987, 0.28612873, 0.31931475, 0.04220394, 0.16093165, 0.22390974, 0.11915915,
        0.3115395,  0.35899726, 0.22190949, 0.57518375, 0.13888834, 0.7753762,  0.4642328,
        0.57055861, 0.21954368, 0.34515455, 0.09486015, 0.40631217, 0.01842281, 0.48770609,
        0.06652815, 0.36023033, 0.42343026, 0.24226256, 0.17348589, 0.44066274, 0.6865865,
        0.17296699, 0.46923906, 0.06921105, 0.3570261,  0.4125829,  0.73165393, 0.15302512,
        0.29499072, 0.33932695, 0.30852377, 0.40762195, 0.40170741, 0.36259529, 0.60848355,
        0.42618036, 0.31721094, 0.02960522, 0.28256637, 0.24389413, 0.2725659,  0.10663581,
        0.27622163, 0.28264219, 0.53652936, 0.09476089, 0.40890986, 0.34848392, 0.32572666,
        0.53076893, 0.11529481, 0.29117745, 0.14625968, 0.8756339,  0.49818122, 0.10656087,
        0.1813329,  0.17664003, 0.21410346, 0.80408043, 0.02315119, 0.27155462, 0.32804728,
        0.13268511, 0.61795473, 0.49703068, 0.41696799, 0.10175809, 0.71028161, 0.29929739,
        0.17377149, 0.76075399, 0.20071237, 0.32632929, 0.36892858, 0.09416146, 0.26656723,
        0.42914796};

    migraphx::shape a_shape{migraphx::shape::float_type, {5, 3, 4, 2}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), al);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(120);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, s));
}

TEST_CASE(sqdiff_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    mm->add_instruction(migraphx::make_op("sqdiff"), l1, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {4, 4, 4};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(sqrt_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {5}};
    std::vector<float> data{1.02481645, 0.85643062, 0.03404123, 0.92791926, 0.10569184};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("sqrt"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return sqrtf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(squeeze_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 3, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("squeeze"), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }
}

TEST_CASE(step_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(2 * 4 * 6);
        std::iota(data.begin(), data.end(), 2);
        migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4, 6}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        auto r  = mm->add_instruction(
            migraphx::make_op("step", {{"axes", {0, 2, 3}}, {"steps", {2, 2, 3}}}), l0);
        mm->add_return({r});
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        migraphx::shape s2{migraphx::shape::float_type, {1, 1, 2, 2}};
        EXPECT(result.get_shape() == s2);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(2 * 4 * 6);
        std::iota(data.begin(), data.end(), 2);
        migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4, 6}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        auto tl = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
        auto r = mm->add_instruction(
            migraphx::make_op("step", {{"axes", {0, 1, 2}}, {"steps", {2, 2, 3}}}), tl);
        mm->add_return({r});
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        migraphx::shape s2{migraphx::shape::float_type, {1, 2, 2, 1}};
        EXPECT(result.get_shape() == s2);
    }
}

TEST_CASE(sub_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    mm->add_instruction(migraphx::make_op("sub"), l1, l2);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2, -2, -2};
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(tan_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-1, 0, 1};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("tan"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return tanf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(tanh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data{-1.0, 2.0, -3.0, 4.0};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("tanh"), l);
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return tanhf(n); });
    EXPECT(migraphx::verify_range(results_vector, gold));
}

TEST_CASE(topk_test)
{
    auto create_program = [](int64_t k, int64_t axis, int largest) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 5}};
        auto data = mm->add_parameter("data", s);
        auto r    = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", axis}, {"k", k}, {"largest", largest}}), data);
        auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        mm->add_return({r0, r1});

        return p;
    };

    auto run_program = [&](int64_t k, int64_t axis, int largest) {
        auto p = create_program(k, axis, largest);
        p.compile(migraphx::ref::target{});
        std::vector<float> data = {
            2.1, 2.3, 2.0, 2.5, 1.9, 3.3, 0.2, 4.5, 0.1, 0.8, 1.0, 4.5, 2.1, 0.8, 1.5};
        migraphx::shape s{migraphx::shape::float_type, {3, 5}};
        migraphx::parameter_map pp;
        pp["data"] = migraphx::argument(s, data.data());
        auto rets  = p.eval(pp);
        std::vector<float> ret_val;
        rets.front().visit([&](auto v) { ret_val.assign(v.begin(), v.end()); });
        std::vector<int64_t> ret_ind;
        rets.back().visit([&](auto v) { ret_ind.assign(v.begin(), v.end()); });

        return std::make_pair(ret_val, ret_ind);
    };

    // case 1
    {
        auto results                = run_program(4, 1, 1);
        std::vector<float> gold_val = {2.5, 2.3, 2.1, 2, 4.5, 3.3, 0.8, 0.2, 4.5, 2.1, 1.5, 1};
        EXPECT(results.first == gold_val);
        std::vector<int64_t> gold_ind = {3, 1, 0, 2, 2, 0, 4, 1, 1, 2, 4, 0};
        EXPECT(results.second == gold_ind);
    }

    // case 2
    {
        auto results                = run_program(4, 1, 0);
        std::vector<float> gold_val = {1.9, 2, 2.1, 2.3, 0.1, 0.2, 0.8, 3.3, 0.8, 1, 1.5, 2.1};
        EXPECT(results.first == gold_val);
        std::vector<int64_t> gold_ind = {4, 2, 0, 1, 3, 1, 4, 0, 3, 0, 4, 2};
        EXPECT(results.second == gold_ind);
    }
}

TEST_CASE(transpose_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2, 2, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);

    {
        migraphx::program p;
        auto* mm                  = p.get_main_module();
        auto l                    = mm->add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> perm = {0, 3, 1, 2};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), l);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();

        result.visit([&](auto output) {
            std::vector<size_t> new_lens = {1, 3, 2, 2};
            EXPECT(bool{output.get_shape().lens() == new_lens});
        });
    }
    {
        migraphx::program p;
        auto* mm                  = p.get_main_module();
        auto l                    = mm->add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> perm = {0, 3, 1, 2};
        auto result =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), l);
        mm->add_instruction(migraphx::make_op("contiguous"), result);
        p.compile(migraphx::ref::target{});
        auto result2 = p.eval({}).back();

        std::vector<float> results_vector(12);
        result2.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
        EXPECT(migraphx::verify_range(results_vector, gold));
    }
}

TEST_CASE(unsqueeze_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l0);
        p.compile(migraphx::ref::target{});
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }
}

TEST_CASE(where_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::bool_type, {3, 3}};
    migraphx::shape sx{migraphx::shape::float_type, {3, 3}};

    std::vector<bool> b{true, true, true, false, false, false, true, false, true};
    std::vector<float> x(9, 1.0);
    std::vector<float> y(9, 2.0);

    auto lb = mm->add_literal(migraphx::literal{sb, b});
    auto lx = mm->add_literal(migraphx::literal{sx, x});
    auto ly = mm->add_literal(migraphx::literal{sx, y});
    auto w  = mm->add_instruction(migraphx::make_op("where"), lb, lx, ly);
    mm->add_return({w});
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<float> gold(9);
    for(int i = 0; i < gold.size(); ++i)
        gold[i] = b[i] ? x[i] : y[i];

    EXPECT(migraphx::verify_range(result_vec, gold));
}

TEST_CASE(where_broadcasted_inputs_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::bool_type, {3, 3}};

    std::vector<bool> b{true, true, true, false, false, false, true, false, true};

    auto lb  = mm->add_literal(migraphx::literal{sb, b});
    auto lx  = mm->add_literal(migraphx::literal(1.0f));
    auto ly  = mm->add_literal(migraphx::literal(2.0f));
    auto mbx = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), lx);
    auto mby = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), ly);
    auto w   = mm->add_instruction(migraphx::make_op("where"), lb, mbx, mby);
    mm->add_return({w});
    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<float> gold(9);
    std::vector<float> x(9, 1.0);
    std::vector<float> y(9, 2.0);
    for(int i = 0; i < gold.size(); ++i)
        gold[i] = b[i] ? x[i] : y[i];

    EXPECT(migraphx::verify_range(result_vec, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
