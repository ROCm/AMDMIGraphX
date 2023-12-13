
#include <onnx_test.hpp>

TEST_CASE(conv_autopad_fail_test)
{
    EXPECT(test::throws([&] { optimize_onnx("conv_autopad_fail_test.onnx"); }));
}
