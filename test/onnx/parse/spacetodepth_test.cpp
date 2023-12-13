
#include <onnx_test.hpp>


TEST_CASE(spacetodepth_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {2, 2, 10, 10}});
    auto tmp1 =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 5, 2, 5, 2}}}), l0);
    auto tmp2 = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 3, 5, 1, 2, 4}}}), tmp1);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 8, 5, 5}}}), tmp2);
    auto prog = optimize_onnx("spacetodepth_test.onnx");
    EXPECT(p == prog);
}


