
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearaveragepool_2d_dilations_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_dilations_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 1, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {108, 112, 124, 127};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
