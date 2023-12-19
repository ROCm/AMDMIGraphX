
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(isinf_double_pos_test)
{
    migraphx::program p = migraphx::parse_onnx("isinf_double_pos_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::double_type, {2, 3}};
    migraphx::parameter_map pp;
    double nan               = std::numeric_limits<double>::quiet_NaN();
    double infinity          = std::numeric_limits<double>::infinity();
    double max               = std::numeric_limits<double>::max();
    double min               = std::numeric_limits<double>::min();
    std::vector<double> data = {-infinity, nan, min, 3.6, max, infinity};
    pp["t1"]                 = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 0, 0, 0, 0, 1};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


