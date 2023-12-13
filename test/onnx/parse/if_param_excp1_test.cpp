
#include <onnx_test.hpp>


TEST_CASE(if_param_excp1_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("if_param_excp1_test.onnx"); }));
}


