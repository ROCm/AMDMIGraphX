
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearleakyrelu_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearSigmoid
    migraphx::program p = migraphx::parse_onnx("qlinearleakyrelu_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x{migraphx::shape::int8_type, {64}};
    std::vector<int8_t> data_x = {
        -128, -124, -120, -116, -112, -108, -104, -100, -96, -92, -88, -84, -80, -76, -72, -68,
        -64,  -60,  -56,  -52,  -48,  -44,  -40,  -36,  -32, -28, -24, -20, -16, -12, -8,  -4,
        0,    4,    8,    12,   16,   20,   24,   28,   32,  36,  40,  44,  48,  52,  56,  60,
        64,   68,   72,   76,   80,   84,   88,   92,   96,  100, 104, 108, 112, 116, 120, 124};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(x, data_x.data());
    auto result = p.eval(pp).back();

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {
        -128, -126, -122, -118, -113, -109, -104, -100, -96, -91, -87, -82, -78, -74, -69, -65,
        -60,  -56,  -52,  -47,  -43,  -38,  -34,  -30,  -25, -21, -16, -12, -8,  -3,  1,   6,
        10,   14,   18,   22,   26,   30,   34,   38,   42,  46,  50,  54,  58,  62,  66,  70,
        74,   78,   82,   86,   90,   94,   98,   102,  106, 110, 114, 118, 122, 126, 127, 127};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
