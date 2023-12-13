
#include <onnx_test.hpp>

TEST_CASE(pad_asym_invalid_pads_error_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("pad_asym_invalid_pads_error_test.onnx"); }));
}
