
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(size_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("size_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {2, 5, 3}};
    std::vector<float> data(30, 1.);
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result      = p.eval(pp).back();
    auto size_result = result.at<int64_t>();
    EXPECT(size_result == int64_t{30});
}


