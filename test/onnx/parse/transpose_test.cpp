
#include <onnx_test.hpp>


TEST_CASE(transpose_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    std::vector<int64_t> perm{0, 3, 1, 2};
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), input);

    auto prog = optimize_onnx("transpose_test.onnx");

    EXPECT(p == prog);
}


