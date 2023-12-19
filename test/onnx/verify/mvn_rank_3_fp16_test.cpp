
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(mvn_rank_3_fp16_test)
{
    using migraphx::half;
    auto result = mvn_test<half>({2, 2, 2}, migraphx::parse_onnx("mvn_rank_3_fp16_test.onnx"));
    std::vector<half> gold{half{-1.342},
                           half{-1.342},
                           half{-0.4473},
                           half{-0.4473},
                           half{0.4473},
                           half{0.4473},
                           half{1.342},
                           half{1.342}};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}
