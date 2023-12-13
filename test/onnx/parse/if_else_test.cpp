
#include <onnx_test.hpp>


TEST_CASE(if_else_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sc{migraphx::shape::bool_type, {1}};
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    std::vector<float> ones(s.elements(), 1.0f);
    std::vector<float> rand = {1.3865, -0.494756, -0.283504, 0.200491, -0.490031, 1.32388};

    auto l1   = mm->add_literal(s, ones);
    auto l2   = mm->add_literal(s, rand);
    auto x    = mm->add_parameter("x", s);
    auto y    = mm->add_parameter("y", s);
    auto cond = mm->add_parameter("cond", sc);

    auto* then_mod = p.create_module("If_5_if");
    auto rt        = then_mod->add_instruction(migraphx::make_op("add"), x, l1);
    then_mod->add_return({rt});

    auto* else_mod = p.create_module("If_5_else");
    auto re        = else_mod->add_instruction(migraphx::make_op("mul"), y, l2);
    else_mod->add_return({re});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("if_else_test.onnx");
    EXPECT(p == prog);
}


