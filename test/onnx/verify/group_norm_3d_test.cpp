
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>


TEST_CASE(group_norm_test)
{
    std::vector<float> scale{1.2, 0.8};
    std::vector<float> bias{0.5, 0.2};
    std::vector<float> result_vector =
        norm_test<float>({1, 4, 2}, scale, bias, migraphx::parse_onnx("group_norm_3d_test.onnx"));
    std::vector<float> gold = {-1.10996256,
                               -0.0366542,
                               1.0366542,
                               2.10996256,
                               -0.87330837,
                               -0.15776947,
                               0.55776947,
                               1.27330837};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


