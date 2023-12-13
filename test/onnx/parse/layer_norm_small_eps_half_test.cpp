
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(layer_norm_small_eps_half_test)
{
    migraphx::program p =
        make_layer_norm({1, 2}, {2}, {1}, 1, true, 1e-7, migraphx::shape::half_type);

    auto prog = optimize_onnx("layer_norm_small_eps_half_test.onnx");
    EXPECT(p == prog);
}
