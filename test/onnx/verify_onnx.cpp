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

TEST_CASE(neg_test)
{
    migraphx::program p = migraphx::parse_onnx("neg_test.onnx");
    p.compile(migraphx::cpu::target{});
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1.0f, 1.3f, -1.2f, 0.0f, -100.f, 200.f};
    migraphx::program::parameter_map pp;
    pp["0"] = migraphx::argument(s, data.data());
    auto result = p.eval({pp}).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-1.0f, -1.3f, 1.2f, 0.0f, 100.f, -200.f};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
