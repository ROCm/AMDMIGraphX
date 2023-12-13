
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(scatter_none_test)
{
    migraphx::program p = create_scatter_program("scatter_none", -2);
    auto prog           = migraphx::parse_onnx("scatter_none_test.onnx");

    EXPECT(p == prog);
}
