
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(layer_norm_2d_axis_zero_test)
{
    migraphx::program p = make_layer_norm({3, 4}, {3, 4}, {0, 1}, 0);

    auto prog = optimize_onnx("layer_norm_2d_axis_zero_test.onnx");
    EXPECT(p == prog);
}


