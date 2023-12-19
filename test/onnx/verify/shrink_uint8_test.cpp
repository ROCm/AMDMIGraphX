
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(shrink_uint8_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_uint8_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::uint8_type, {3, 3}};
    std::vector<uint8_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold = {0, 0, 0, 0, 0, 10, 11, 12, 13};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


