
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearaveragepool_2d_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {84,  -73, 117, -2,  -97, 72,   67,  27,   1,  -44,  110, 51,
                                  9,   7,   58,  113, -34, 34,   124, -20,  6,  66,   68,  98,
                                  31,  -84, 25,  101, -69, -100, -68, 116,  33, -121, 78,  49,
                                  102, -86, 65,  69,  -87, -89,  16,  -125, 51, -54,  -86, 79};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {4,   127, 127, -41,  127, 127, -6,   125,  127,
                                76,  127, 127, 32,   78,  127, -128, -128, 127,
                                -44, -37, 127, -117, -62, 37,  -128, -128, -81};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
