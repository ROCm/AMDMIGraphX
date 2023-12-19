
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>


TEST_CASE(group_norm_half_test)
{
    using migraphx::half;
    std::vector<half> scale{half{1.2}, half{0.8}};
    std::vector<half> bias{half{0.5}, half{0.2}};
    std::vector<half> result_vector = norm_test<half>(
        {1, 4, 2}, scale, bias, migraphx::parse_onnx("group_norm_3d_half_test.onnx"));
    std::vector<half> gold = {half{-1.10996256},
                              half{-0.0366542},
                              half{1.0366542},
                              half{2.10996256},
                              half{-0.87330837},
                              half{-0.15776947},
                              half{0.55776947},
                              half{1.27330837}};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


