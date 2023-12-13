
#include <onnx_test.hpp>

TEST_CASE(randomnormallike_type_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("randomnormallike_type_error_test.onnx"); }));
}
