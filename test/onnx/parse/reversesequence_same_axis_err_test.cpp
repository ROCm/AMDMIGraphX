
#include <onnx_test.hpp>

TEST_CASE(reversesequence_same_axis_err_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("reversesequence_same_axis_err_test.onnx"); }));
}
