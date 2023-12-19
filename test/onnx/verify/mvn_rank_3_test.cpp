
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(mvn_rank_3_test)
{
    auto result = mvn_test({2, 2, 2}, migraphx::parse_onnx("mvn_rank_3_test.onnx"));
    std::vector<float> gold{-1.34164079,
                            -1.34164079,
                            -0.4472136,
                            -0.4472136,
                            0.4472136,
                            0.4472136,
                            1.34164079,
                            1.34164079};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}
