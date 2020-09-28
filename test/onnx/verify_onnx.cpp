#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

TEST_CASE(instance_norm_test)
{
    migraphx::program p = migraphx::parse_onnx("instance_norm_val_test.onnx");

    p.compile(migraphx::cpu::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vector(9);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-1.54919,
                               -1.16189,
                               -0.774596,
                               -0.387298,
                               0,
                               0.387298,
                               0.774596,
                               1.16189,
                               1.54919,
                               -2.09838,
                               -1.32379,
                               -0.549192,
                               0.225404,
                               1,
                               1.7746,
                               2.54919,
                               3.32379,
                               4.09838};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(instance_norm_3d_test)
{
    migraphx::program p = migraphx::parse_onnx("instance_norm_val_3d_test.onnx");

    p.compile(migraphx::cpu::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vector(16);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-1.52752,
                               -1.09109,
                               -0.654653,
                               -0.218218,
                               0.218218,
                               0.654653,
                               1.09109,
                               1.52752,
                               -2.05505,
                               -1.18218,
                               -0.309306,
                               0.563565,
                               1.43644,
                               2.30931,
                               3.18218,
                               4.05505};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(averagepool_notset_test)
{
    auto p = migraphx::parse_onnx("averagepool_notset_test.onnx");
    p.compile(migraphx::cpu::target{});
    std::vector<float> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::program::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {12};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(averagepool_nt_cip_test)
{
    auto p = migraphx::parse_onnx("averagepool_nt_cip_test.onnx");
    p.compile(migraphx::cpu::target{});
    std::vector<float> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::program::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {8.33333};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(gather_elements)
{
    migraphx::program p = migraphx::parse_onnx("gather_elements_axis0_test.onnx");
    p.compile(migraphx::cpu::target{});
    migraphx::shape s_data{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data = {
        0.25, 0.75, 0.9375, 0.4375, 0.6875, 0.5625, -0.875, 0.1875, -0.125, 0.5, -0.9375, -0.0625};

    migraphx::shape s_ind{migraphx::shape::int32_type, {2, 3}};
    std::vector<int> ind = {2, 1, 2, 0, 1, 0};

    migraphx::program::parameter_map pp;
    pp["data"]    = migraphx::argument(s_data, data.data());
    pp["indices"] = migraphx::argument(s_ind, ind.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-0.125, 0.5625, -0.9375, 0.25, 0.5625, 0.9375};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(selu_test)
{
    migraphx::program p = migraphx::parse_onnx("selu_test.onnx");
    p.compile(migraphx::cpu::target{});

    migraphx::shape xs{migraphx::shape::double_type, {2, 3}};
    std::vector<double> x_data = {1.1, 2.1, 0.0, -1.3, -5.3, 12.0};

    migraphx::program::parameter_map pp;
    pp["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.55, 1.05, 0, -0.10912, -0.149251, 6};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(where_test)
{
    migraphx::program p = migraphx::parse_onnx("where_test.onnx");
    p.compile(migraphx::cpu::target{});

    migraphx::shape c_shape{migraphx::shape::bool_type, {2}};
    std::vector<int8_t> c_data = {1, 0};

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 2, 2}};
    std::vector<float> x_data(8, 1.0f);

    migraphx::shape y_shape{migraphx::shape::float_type, {2, 1, 2, 2}};
    std::vector<float> y_data(8, 2.0f);

    migraphx::program::parameter_map pp;
    pp["c"] = migraphx::argument(c_shape, c_data.data());
    pp["x"] = migraphx::argument(x_shape, x_data.data());
    pp["y"] = migraphx::argument(y_shape, y_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
