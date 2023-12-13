
#include <onnx_test.hpp>


TEST_CASE(flatten_nonstd_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 5, 4}});
    auto l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), l0);
    auto l2 = mm->add_instruction(migraphx::make_op("contiguous"), l1);
    mm->add_instruction(migraphx::make_op("flatten", {{"axis", 2}}), l2);
    auto l3 = mm->add_instruction(migraphx::make_op("contiguous"), l1);
    mm->add_instruction(migraphx::make_op("flatten", {{"axis", 1}}), l3);
    auto prog = optimize_onnx("flatten_nonstd_test.onnx");

    EXPECT(p == prog);
}


