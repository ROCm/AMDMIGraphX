
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearmul_bcast_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearMul
    migraphx::program p = migraphx::parse_onnx("qlinearmul_bcast_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::int8_type, {64}};
    std::vector<int8_t> data_a = {-64, -62, -60, -58, -56, -54, -52, -50, -48, -46, -44, -42, -40,
                                  -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -18, -16, -14,
                                  -12, -10, -8,  -6,  -4,  -2,  0,   2,   4,   6,   8,   10,  12,
                                  14,  16,  18,  20,  22,  24,  26,  28,  30,  32,  34,  36,  38,
                                  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62};

    migraphx::shape b{migraphx::shape::int8_type, {1, 1, 64}};
    std::vector<int8_t> data_b = {96, 94,  92,  90,  88,  86,  84,  82,  80,  78,  76,  74, 72,
                                  70, 68,  66,  64,  62,  60,  58,  56,  54,  52,  50,  48, 46,
                                  44, 42,  40,  38,  36,  34,  32,  30,  28,  26,  24,  22, 20,
                                  18, 16,  14,  12,  10,  8,   6,   4,   2,   0,   -2,  -4, -6,
                                  -8, -10, -12, -14, -16, -18, -20, -22, -24, -26, -28, -30};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {-128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -118,
                                -109, -101, -93,  -86,  -78,  -70,  -63,  -56,  -49,  -42,  -35,
                                -28,  -21,  -15,  -9,   -2,   4,    10,   15,   21,   27,   32,
                                37,   42,   47,   52,   57,   62,   66,   70,   75,   79,   83,
                                86,   90,   94,   97,   100,  103,  106,  109,  112,  115,  117,
                                119,  122,  124,  126,  127,  127,  127,  127,  127};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
