
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_if_lp : verify_program<test_if_lp>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        migraphx::shape s{migraphx::shape::float_type, {5}};
        auto cond = mm->add_parameter("cond", cond_s);
        auto x    = mm->add_parameter("x", s);

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {1, 2, 3, 4, 5};
        auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
        then_mod->add_return({l1, x});

        auto* else_mod           = p.create_module("If_0_else");
        std::vector<float> data2 = {5, 4, 3, 2, 1};
        auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
        auto s2                  = else_mod->add_instruction(migraphx::make_op("add"), x, l2);
        else_mod->add_return({s2, l2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r0  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        auto r1  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), ret);
        mm->add_return({r0, r1});

        return p;
    }
};
