
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(layer_norm_2d_axis_one_test)
{
    migraphx::program p = make_layer_norm({3, 4}, {4}, {1}, 1);

    auto prog = optimize_onnx("layer_norm_2d_axis_one_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(layer_norm_2d_axis_minus_one_test)
{
    migraphx::program p = make_layer_norm({3, 4}, {4}, {1}, 1);

    auto prog = optimize_onnx("layer_norm_2d_axis_one_test.onnx");
    EXPECT(p == prog);
}
