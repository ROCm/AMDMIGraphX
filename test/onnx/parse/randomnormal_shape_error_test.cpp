
#include <onnx_test.hpp>

TEST_CASE(randomnormal_shape_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("randomnormal_shape_error_test.onnx"); }));
}
