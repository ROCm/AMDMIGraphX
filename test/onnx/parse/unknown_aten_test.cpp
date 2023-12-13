
#include <onnx_test.hpp>

TEST_CASE(unknown_aten_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("unknown_aten_test.onnx"); }));
}
