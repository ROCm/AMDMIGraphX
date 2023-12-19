
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(if_then_test_inlined)
{
    migraphx::program p = migraphx::parse_onnx("if_then_test_inlined.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s_data{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_data, data.data());
    pp["y"] = migraphx::argument(s_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.0625, 1.75, 0.9375, 1.125, 0.875, 0.4375};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


