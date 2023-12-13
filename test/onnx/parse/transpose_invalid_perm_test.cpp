
#include <onnx_test.hpp>

TEST_CASE(transpose_invalid_perm_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("transpose_invalid_perm_test.onnx"); }));
}
