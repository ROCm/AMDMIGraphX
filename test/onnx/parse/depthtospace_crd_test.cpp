
#include <onnx_test.hpp>

TEST_CASE(depthtospace_crd_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {2, 8, 5, 5}});
    auto tmp1 =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), l0);
    auto tmp2 = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), tmp1);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 10, 10}}}), tmp2);
    auto prog = optimize_onnx("depthtospace_crd_test.onnx");
    EXPECT(p == prog);
}
