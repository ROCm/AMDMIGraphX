
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(shrink_int8_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_int8_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::int8_type, {3, 3}};
    std::vector<int8_t> data{-4, -3, -2, -1, 0, 1, 2, 3, 4};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int8_t> gold = {-2, -1, 0, 0, 0, 0, 0, 1, 2};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
