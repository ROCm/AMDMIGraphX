
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>


TEST_CASE(mvn_rank_2_test)
{
    auto result = mvn_test({2, 2}, migraphx::parse_onnx("mvn_rank_2_test.onnx"));
    std::vector<float> gold{-1, 1, -1, 1};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}


