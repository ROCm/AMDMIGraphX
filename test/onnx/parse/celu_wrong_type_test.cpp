
#include <onnx_test.hpp>

TEST_CASE(celu_wrong_type_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("celu_wrong_type_test.onnx"); }));
}
