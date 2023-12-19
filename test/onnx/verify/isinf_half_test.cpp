
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(isinf_half_test)
{
    migraphx::program p = migraphx::parse_onnx("isinf_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::half_type, {2, 3}};
    migraphx::parameter_map pp;
    migraphx::half nan               = std::numeric_limits<migraphx::half>::quiet_NaN();
    migraphx::half infinity          = std::numeric_limits<migraphx::half>::infinity();
    migraphx::half max               = std::numeric_limits<migraphx::half>::max();
    migraphx::half min               = std::numeric_limits<migraphx::half>::min();
    migraphx::half val               = migraphx::half(3.6);
    std::vector<migraphx::half> data = {-infinity, nan, min, val, max, infinity};
    pp["t1"]                         = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 0, 0, 0, 0, 1};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


