
#include <onnx_test.hpp>

TEST_CASE(spacetodepth_nondivisibility_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("spacetodepth_nondivisibility_test.onnx"); }));
}
