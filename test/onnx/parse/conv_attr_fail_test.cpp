
#include <onnx_test.hpp>

TEST_CASE(conv_attr_fail_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("conv_attr_fail_test.onnx"); }));
}
