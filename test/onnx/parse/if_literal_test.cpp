
#include <onnx_test.hpp>


TEST_CASE(if_literal_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond = mm->add_parameter("cond", cond_s);

    migraphx::shape s{migraphx::shape::float_type, {5}};

    auto* then_mod           = p.create_module("If_1_if");
    std::vector<float> data1 = {1, 2, 3, 4, 5};
    auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
    then_mod->add_literal({});
    then_mod->add_return({l1});

    auto* else_mod           = p.create_module("If_1_else");
    std::vector<float> data2 = {5, 4, 3, 2, 1};
    auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
    else_mod->add_literal({});
    else_mod->add_return({l2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("if_literal_test.onnx");
    EXPECT(p == prog);
}


