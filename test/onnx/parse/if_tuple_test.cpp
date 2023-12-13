
#include <onnx_test.hpp>

TEST_CASE(if_tuple_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {1}};
    auto l1 = mm->add_literal(migraphx::literal(sd, {1}));
    auto l2 = mm->add_literal(migraphx::literal(sd, {2}));
    auto l3 = mm->add_literal(migraphx::literal(sd, {3}));
    migraphx::shape sx{migraphx::shape::float_type, {1, 4}};
    migraphx::shape sy{migraphx::shape::float_type, {3, 4}};
    migraphx::shape sc{migraphx::shape::bool_type};
    auto cond = mm->add_parameter("cond", sc);
    auto x    = mm->add_parameter("x", sx);
    auto y    = mm->add_parameter("y", sy);

    auto* then_mod = p.create_module("If_6_if");
    auto m1 =
        then_mod->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), l1);
    auto add0 = then_mod->add_instruction(migraphx::make_op("add"), x, m1);
    auto m2 =
        then_mod->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), l2);
    auto mul0 = then_mod->add_instruction(migraphx::make_op("mul"), y, m2);
    then_mod->add_return({add0, mul0});

    auto* else_mod = p.create_module("If_6_else");
    auto me1 =
        else_mod->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), l3);
    auto mul1 = else_mod->add_instruction(migraphx::make_op("mul"), x, me1);
    auto me2 =
        else_mod->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), l3);
    auto add1 = else_mod->add_instruction(migraphx::make_op("add"), y, me2);
    else_mod->add_return({mul1, add1});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r0  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    auto r1  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), ret);
    mm->add_return({r0, r1});

    auto prog = migraphx::parse_onnx("if_tuple_test.onnx");
    EXPECT(p == prog);
}
