
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(group_norm_3d_test)
{
    migraphx::program p = make_group_norm(
        {1, 4, 2}, {2}, {2}, {1, 2, 2, 2}, {2, 3}, 1e-5f, migraphx::shape::float_type);
    auto prog = optimize_onnx("group_norm_3d_test.onnx");
    EXPECT(p == prog);
}
