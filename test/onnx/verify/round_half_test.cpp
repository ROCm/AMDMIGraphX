
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(round_half_test)
{
    migraphx::program p = migraphx::parse_onnx("round_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape xs{migraphx::shape::half_type, {4, 4}};
    std::vector<float> tmp = {-3.51,
                              -3.5,
                              -3.49,
                              -2.51,
                              -2.50,
                              -2.49,
                              -1.6,
                              -1.5,
                              -0.51,
                              -0.5,
                              0.5,
                              0.6,
                              2.4,
                              2.5,
                              3.5,
                              4.5};
    std::vector<migraphx::half> data{tmp.cbegin(), tmp.cend()};
    migraphx::parameter_map param_map;
    param_map["x"] = migraphx::argument(xs, data.data());

    auto result = p.eval(param_map).back();

    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {-4.0, -4.0, -3.0, -3.0, -2.0, -2.0, -2.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 4.0};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
