
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(scatter_add_test)
{
    migraphx::program p = create_scatter_program("scatter_add", -2);
    auto prog           = migraphx::parse_onnx("scatter_add_test.onnx");

    EXPECT(p == prog);
}
