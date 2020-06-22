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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
