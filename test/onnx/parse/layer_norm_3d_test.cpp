
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(layer_norm_3d_test)
{
    migraphx::program p = make_layer_norm({1, 4, 2}, {2}, {2}, 2);

    auto prog = optimize_onnx("layer_norm_3d_test.onnx");
    EXPECT(p == prog);
}


