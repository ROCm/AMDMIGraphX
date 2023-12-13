
#include <onnx_test.hpp>

TEST_CASE(mvn_axes_rank_too_small_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("mvn_axes_rank_too_small_test.onnx"); }));
}
