
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(resize_upsample_linear_ac_test)
{
    auto p    = create_upsample_linear_prog();
    auto prog = migraphx::parse_onnx("resize_upsample_linear_ac_test.onnx");
    EXPECT(p == prog);
}
