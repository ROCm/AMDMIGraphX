
#include <onnx_test.hpp>


TEST_CASE(if_param_excp_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("if_param_excp_test.onnx"); }));
}


