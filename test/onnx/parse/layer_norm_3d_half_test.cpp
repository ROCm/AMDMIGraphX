
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(layer_norm_3d_half_test)
{
    migraphx::program p =
        make_layer_norm({1, 4, 2}, {2}, {2}, 2, false, 1e-5f, migraphx::shape::half_type);

    auto prog = optimize_onnx("layer_norm_3d_half_test.onnx");
    EXPECT(p == prog);
}


