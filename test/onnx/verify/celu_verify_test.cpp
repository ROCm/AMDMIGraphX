
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(celu_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("celu_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {-5.5, 2.0, 100., 7.0, 0., -1.};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s, data.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold(6);
    float alpha = 0.5;
    std::transform(data.begin(), data.end(), gold.begin(), [&](auto x) {
        return std::max(0.0f, x) + std::min(0.0f, alpha * std::expm1(x / alpha));
    });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


