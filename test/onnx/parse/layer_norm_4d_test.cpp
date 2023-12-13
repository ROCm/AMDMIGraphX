
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(layer_norm_4d_test)
{
    migraphx::program p = make_layer_norm({3, 3, 3, 3}, {3}, {3}, 3);

    auto prog = optimize_onnx("layer_norm_4d_test.onnx");
    EXPECT(p == prog);
}


