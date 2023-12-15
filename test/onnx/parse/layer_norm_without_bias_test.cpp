
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(layer_norm_without_bias_test)
{
    migraphx::program p = make_layer_norm({1, 2}, {2}, {1}, 1, true);

    auto prog = optimize_onnx("layer_norm_without_bias_test.onnx");
    EXPECT(p == prog);
}
