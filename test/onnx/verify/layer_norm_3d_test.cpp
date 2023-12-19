
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>


TEST_CASE(layer_norm_test)
{
    std::vector<float> scale{1.2, 0.8};
    std::vector<float> bias{0.5, 0.2};
    std::vector<float> result_vector =
        norm_test<float>({1, 4, 2}, scale, bias, migraphx::parse_onnx("layer_norm_3d_test.onnx"));
    std::vector<float> gold = {-0.69997597,
                               0.99998398,
                               -0.69997597,
                               0.99998398,
                               -0.69997597,
                               0.99998398,
                               -0.69997597,
                               0.99998398};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


