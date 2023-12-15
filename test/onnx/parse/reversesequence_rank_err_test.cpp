
#include <onnx_test.hpp>

TEST_CASE(reversesequence_rank_err_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("reversesequence_rank_err_test.onnx"); }));
}
