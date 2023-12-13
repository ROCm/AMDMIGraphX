
#include <onnx_test.hpp>


TEST_CASE(multinomial_dtype_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("multinomial_dtype_error_test.onnx"); }));
}


