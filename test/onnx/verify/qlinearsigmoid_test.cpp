
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(qlinearsigmoid_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearSigmoid
    migraphx::program p = migraphx::parse_onnx("qlinearsigmoid_test.onnx");
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

    std::vector<int8_t> gold = {-128, -127, -127, -127, -127, -127, -126, -126, -126, -125, -125,
                                -124, -123, -122, -120, -119, -117, -114, -112, -108, -104, -99,
                                -94,  -87,  -80,  -71,  -62,  -51,  -39,  -27,  -13,  1,    15,
                                29,   43,   56,   69,   81,   92,   101,  110,  117,  124,  127,
                                127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,
                                127,  127,  127,  127,  127,  127,  127,  127,  127};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


