
#include <onnx_test.hpp>

TEST_CASE(spacetodepth_simple_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 2, 4, 6}});
    auto tmp1 =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 2}}}), l0);
    auto tmp2 = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 3, 5, 1, 2, 4}}}), tmp1);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 8, 2, 3}}}), tmp2);
    auto prog = optimize_onnx("spacetodepth_simple_test.onnx");
    EXPECT(p == prog);
}
