
#include <onnx_test.hpp>


TEST_CASE(selu_test)
{
    migraphx::program p;
    auto* mm                      = p.get_main_module();
    std::vector<std::size_t> lens = {2, 3};
    migraphx::shape s{migraphx::shape::double_type, lens};
    auto x = mm->add_parameter("x", s);

    migraphx::shape ls{migraphx::shape::double_type, {1}};
    auto la   = mm->add_literal({ls, {0.3}});
    auto lg   = mm->add_literal({ls, {0.25}});
    auto mbla = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), la);
    auto mblg = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), lg);

    auto sign_x = mm->add_instruction(migraphx::make_op("sign"), x);
    auto exp_x  = mm->add_instruction(migraphx::make_op("exp"), x);

    auto mlax  = mm->add_instruction(migraphx::make_op("mul"), mbla, exp_x);
    auto smlax = mm->add_instruction(migraphx::make_op("sub"), mlax, mbla);

    auto item1 = mm->add_instruction(migraphx::make_op("add"), smlax, x);
    auto item2 = mm->add_instruction(migraphx::make_op("sub"), smlax, x);

    auto sitem2 = mm->add_instruction(migraphx::make_op("mul"), sign_x, item2);
    auto item12 = mm->add_instruction(migraphx::make_op("sub"), item1, sitem2);
    auto r      = mm->add_instruction(migraphx::make_op("mul"), item12, mblg);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("selu_test.onnx");

    EXPECT(p == prog);
}


