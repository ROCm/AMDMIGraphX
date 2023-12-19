
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearaveragepool_2d_same_upper_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_same_upper_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {-61, 102,  -6,  61,  -34,  6,    -13, -38, -26, 105,  36,  116,
                                  -62, 31,   113, 85,  126,  -52,  80,  38,  115, -89,  -35, 67,
                                  69,  -116, 11,  -47, -120, 120,  39,  96,  29,  5,    -89, 40,
                                  58,  51,   -99, -77, -12,  -107, 76,  -13, 126, -112, -64, -57};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {
        -58, -20,  -62,  -41,  -38, 3,    -14,  14,   -40,  78,   111, 127,  -95, 80,   127,  106,
        -14, -112, 11,   41,   -74, -128, -66,  -44,  -88,  -37,  -14, -15,  -64, 95,   71,   127,
        8,   -128, -128, -101, -69, -104, -120, -128, -116, -128, -93, -128, -50, -128, -128, -128};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
