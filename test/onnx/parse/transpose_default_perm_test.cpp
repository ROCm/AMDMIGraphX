
#include <onnx_test.hpp>

TEST_CASE(transpose_default_perm_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 3}});
    std::vector<int64_t> perm{3, 2, 1, 0};
    auto r = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), input);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("transpose_default_perm_test.onnx");

    EXPECT(p == prog);
}
