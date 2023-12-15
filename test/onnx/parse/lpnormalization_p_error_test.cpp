
#include <onnx_test.hpp>


TEST_CASE(lpnormalization_p_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("lpnormalization_p_error_test.onnx"); }));
}


