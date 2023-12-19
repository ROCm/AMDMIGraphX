
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(qlinearglobalavgpool_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
    // #com.microsoft.QLinearGlobalAveragePool

    migraphx::program p = migraphx::parse_onnx("qlinearglobalavgpool_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sh_x{migraphx::shape::uint8_type, {1, 3, 4, 4}};
    std::vector<uint8_t> data_x = {160, 156, 152, 148, 144, 140, 136, 132, 124, 120, 116, 112,
                                   108, 104, 100, 96,  64,  72,  80,  88,  96,  104, 112, 120,
                                   136, 144, 152, 160, 168, 176, 184, 192, 120, 121, 122, 123,
                                   124, 125, 126, 127, 129, 130, 131, 132, 133, 134, 135, 136};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sh_x, data_x.data());

    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {64, 64, 64};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


