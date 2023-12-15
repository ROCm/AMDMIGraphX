
#include <onnx_test.hpp>

TEST_CASE(mod_test_half)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("mod_test_half.onnx"); }));
}
