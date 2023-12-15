
#include <onnx_test.hpp>

TEST_CASE(reversesequence_batch_axis_err_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("reversesequence_batch_axis_err_test.onnx"); }));
}
