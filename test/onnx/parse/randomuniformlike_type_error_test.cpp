
#include <onnx_test.hpp>


TEST_CASE(randomuniformlike_type_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("randomuniformlike_type_error_test.onnx"); }));
}


