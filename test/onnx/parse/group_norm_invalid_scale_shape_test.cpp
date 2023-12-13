
#include <onnx_test.hpp>

TEST_CASE(group_norm_invalid_scale_shape_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("group_norm_invalid_scale_shape_test.onnx"); }));
}
