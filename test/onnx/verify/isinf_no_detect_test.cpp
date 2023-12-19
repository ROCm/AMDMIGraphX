
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(isinf_no_detect_test)
{
    migraphx::program p = migraphx::parse_onnx("isinf_no_detect_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::parameter_map pp;
    float nan                = std::numeric_limits<float>::quiet_NaN();
    float infinity           = std::numeric_limits<float>::infinity();
    float max                = std::numeric_limits<float>::max();
    float min                = std::numeric_limits<float>::min();
    std::vector<double> data = {-infinity, nan, min, 3.6, max, infinity};
    pp["t1"]                 = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


