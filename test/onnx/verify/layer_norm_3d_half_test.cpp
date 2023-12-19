
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(layer_norm_half_test)
{
    using migraphx::half;
    std::vector<half> scale{half{1.2}, half{0.8}};
    std::vector<half> bias{half{0.5}, half{0.2}};
    std::vector<half> result_vector = norm_test<half>(
        {1, 4, 2}, scale, bias, migraphx::parse_onnx("layer_norm_3d_half_test.onnx"));
    std::vector<half> gold = {half{-0.69997597},
                              half{0.99998398},
                              half{-0.69997597},
                              half{0.99998398},
                              half{-0.69997597},
                              half{0.99998398},
                              half{-0.69997597},
                              half{0.99998398}};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
