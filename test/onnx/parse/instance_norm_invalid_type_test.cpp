
#include <onnx_test.hpp>

TEST_CASE(instance_norm_invalid_type_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("instance_norm_invalid_type_test.onnx"); }));
}
