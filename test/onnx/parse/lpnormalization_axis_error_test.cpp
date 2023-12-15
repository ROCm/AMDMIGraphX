
#include <onnx_test.hpp>


TEST_CASE(lpnormalization_axis_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("lpnormalization_axis_error_test.onnx"); }));
}


