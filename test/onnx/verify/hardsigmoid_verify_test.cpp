
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(hardsigmoid_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("hardsigmoid_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {2, 5}};
    std::vector<float> data = {-10.0, -2.5, -1.0, -0.5, 0, 1.0, 2.0, 2.5, 2.6, 100.0};

    float alpha = 0.2;
    float beta  = 0.5;
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold(10);
    std::transform(data.begin(), data.end(), gold.begin(), [&](auto x) {
        return std::max(0.0f, std::min(x * alpha + beta, 1.0f));
    });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


