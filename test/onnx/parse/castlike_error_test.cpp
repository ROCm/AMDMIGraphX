
#include <onnx_test.hpp>


TEST_CASE(castlike_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("castlike_error_test.onnx"); }));
}


