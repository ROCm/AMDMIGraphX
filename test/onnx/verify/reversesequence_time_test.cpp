
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(reversesequence_time_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("reversesequence_time_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape xs{migraphx::shape::float_type, {4, 4}};
    std::vector<float> x_data = {
        0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0};
    migraphx::parameter_map param_map;
    param_map["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        3.0, 6.0, 9.0, 12.0, 2.0, 5.0, 8.0, 13.0, 1.0, 4.0, 10.0, 14.0, 0.0, 7.0, 11.0, 15.0};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


