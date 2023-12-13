
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(upsample_linear_test)
{
    auto p    = create_upsample_linear_prog();
    auto prog = migraphx::parse_onnx("upsample_linear_test.onnx");
    EXPECT(p == prog);
}


