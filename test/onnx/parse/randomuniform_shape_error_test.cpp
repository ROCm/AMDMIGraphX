
#include <onnx_test.hpp>

TEST_CASE(randomuniform_shape_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("randomuniform_shape_error_test.onnx"); }));
}
