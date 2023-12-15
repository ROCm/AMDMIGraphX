
#include <onnx_test.hpp>

TEST_CASE(if_pl_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    migraphx::shape xs{migraphx::shape::float_type, {2, 3}};
    migraphx::shape ys{migraphx::shape::float_type, {3, 3}};
    std::vector<float> datax = {1, 2, 3, 4, 5, 6};
    std::vector<float> datay = {8, 7, 6, 5, 4, 3, 2, 1, 0};

    auto lx   = mm->add_literal(migraphx::literal(xs, datax));
    auto ly   = mm->add_literal(migraphx::literal(ys, datay));
    auto cond = mm->add_parameter("cond", cond_s);
    auto x    = mm->add_parameter("x", xs);
    auto y    = mm->add_parameter("y", ys);

    auto* then_mod = p.create_module("If_5_if");
    auto l1        = then_mod->add_literal(migraphx::literal(ys, datay));
    auto a1        = then_mod->add_instruction(migraphx::make_op("add"), x, lx);
    then_mod->add_return({a1, l1});

    auto* else_mod = p.create_module("If_5_else");
    auto l2        = else_mod->add_literal(migraphx::literal(xs, datax));
    auto a2        = else_mod->add_instruction(migraphx::make_op("mul"), y, ly);
    else_mod->add_return({l2, a2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), ret);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("if_pl_test.onnx");
    EXPECT(p == prog);
}
