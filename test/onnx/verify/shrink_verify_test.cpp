
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(shrink_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::half_type, {5}};
    std::vector<float> tmp = {-10.0, -5.0, 0.0, 5.0, 10.0};
    std::vector<migraphx::half> data{tmp.cbegin(), tmp.cend()};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    tmp = {-9.0, -4.0, 1.0, 4.0, 9.0};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
