
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(shrink_soft_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_soft_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {5}};
    std::vector<float> data{-2, -1, 0, 1, 2};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.5, 0, 0, 0, 0.5};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
