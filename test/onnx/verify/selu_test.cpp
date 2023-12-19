
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(selu_test)
{
    migraphx::program p = migraphx::parse_onnx("selu_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape xs{migraphx::shape::double_type, {2, 3}};
    std::vector<double> x_data = {1.1, 2.1, 0.0, -1.3, -5.3, 12.0};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.55, 1.05, 0, -0.10912, -0.149251, 6};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


