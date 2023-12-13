
#include <onnx_test.hpp>


TEST_CASE(pad_reflect_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {0, 2, 0, 1}});
    auto l1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 1}}, {"ends", {2, 2}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 0}}, {"ends", {2, 1}}}), l0);
    auto l3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 0}}, {"ends", {2, 1}}}), l0);
    auto r = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l2, l1, l0, l3);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("pad_reflect_test.onnx");

    EXPECT(p == prog);
}


