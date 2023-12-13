
#include <onnx_test.hpp>

TEST_CASE(scatternd_invalid_reduction_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("scatternd_invalid_reduction_test.onnx"); }));
}
