
#include <onnx_test.hpp>


TEST_CASE(isinf_double_pos_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {2, 3}};
    auto t1     = mm->add_parameter("t1", s);
    auto is_inf = mm->add_instruction(migraphx::make_op("isinf"), t1);
    auto zero_l = mm->add_literal(migraphx::literal{migraphx::shape::double_type, {0}});
    auto mb_zero =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), zero_l);

    auto is_neg = mm->add_instruction(migraphx::make_op("greater"), t1, mb_zero);
    if(is_neg->get_shape().type() != migraphx::shape::bool_type)
    {
        is_neg = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), is_neg);
    }
    auto ret = mm->add_instruction(migraphx::make_op("logical_and"), is_inf, is_neg);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("isinf_double_pos_test.onnx");
    EXPECT(p == prog);
}


