
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(group_norm_4d_half_test)
{
    migraphx::program p = make_group_norm(
        {1, 4, 3, 3}, {2}, {2}, {1, 2, 2, 3, 3}, {2, 3, 4}, 1e-5f, migraphx::shape::half_type);
    auto prog = optimize_onnx("group_norm_4d_half_test.onnx");
    EXPECT(p == prog);
}


