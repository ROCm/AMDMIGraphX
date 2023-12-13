
#include <onnx_test.hpp>


TEST_CASE(batch_norm_invalid_bias_rank)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("batch_norm_invalid_bias_rank.onnx"); }));
}


