
#include <onnx_test.hpp>


TEST_CASE(celu_zero_alpha_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("celu_zero_alpha_test.onnx"); }));
}


