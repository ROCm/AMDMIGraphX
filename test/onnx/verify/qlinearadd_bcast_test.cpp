
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearadd_bcast_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearAdd
    migraphx::program p = migraphx::parse_onnx("qlinearadd_bcast_test.onnx");
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

    std::vector<int8_t> gold = {-64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64,
                                -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64,
                                -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64,
                                -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64,
                                -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
