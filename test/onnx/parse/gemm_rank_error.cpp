
#include <onnx_test.hpp>

TEST_CASE(gemm_rank_error)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("gemm_rank_error.onnx"); }));
}
