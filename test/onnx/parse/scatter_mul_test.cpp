
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(scatter_mul_test)
{
    migraphx::program p = create_scatter_program("scatter_mul", -2);
    auto prog           = migraphx::parse_onnx("scatter_mul_test.onnx");

    EXPECT(p == prog);
}
