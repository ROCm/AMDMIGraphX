
#include <onnx_test.hpp>

TEST_CASE(spacetodepth_invalid_blocksize)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("spacetodepth_invalid_blocksize_test.onnx"); }));
}
