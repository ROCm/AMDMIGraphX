
#include <onnx_test.hpp>

TEST_CASE(if_then_else_multi_output_shapes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sc{migraphx::shape::bool_type, {1}};

    migraphx::shape s{migraphx::shape::float_type, {2, 3, 1}};
    migraphx::shape s_trail{migraphx::shape::float_type, {2, 3, 1}};
    std::vector<float> ones(s.elements(), 1.0f);

    auto l1                 = mm->add_literal(s_trail, ones);
    std::vector<float> rand = {-0.753997, 0.707831, -0.865795, 2.49574, 0.464937, -0.168745};
    auto l2                 = mm->add_literal(s, rand);
    auto x                  = mm->add_parameter("x", s_trail);
    auto y                  = mm->add_parameter("y", s);
    auto cond               = mm->add_parameter("cond", sc);

    auto* then_mod = p.create_module("If_5_if");
    auto rt        = then_mod->add_instruction(migraphx::make_op("add"), x, l1);
    auto rt2       = then_mod->add_instruction(migraphx::make_op("add"), x, x);
    then_mod->add_return({rt, rt2});

    auto* else_mod = p.create_module("If_5_else");
    auto re        = else_mod->add_instruction(migraphx::make_op("mul"), y, l2);
    auto re2       = else_mod->add_instruction(migraphx::make_op("sub"), y, l2);
    else_mod->add_return({re, re2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r1  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    auto r2  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), ret);
    mm->add_return({r1, r2});

    auto prog = migraphx::parse_onnx("if_then_else_multi_output_shapes_test.onnx");
    EXPECT(p == prog);
}
