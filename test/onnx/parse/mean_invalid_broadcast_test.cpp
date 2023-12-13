
#include <onnx_test.hpp>

TEST_CASE(mean_invalid_broadcast_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("mean_invalid_broadcast_test.onnx"); }));
}
