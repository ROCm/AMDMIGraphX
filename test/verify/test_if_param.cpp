
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_if_param : verify_program<test_if_param>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", cond_s);
        migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
        auto x = mm->add_parameter("x", ds);
        auto y = mm->add_parameter("y", ds);

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {0.384804, -1.77948, -0.453775, 0.477438, -1.06333, -1.12893};
        auto l1                  = then_mod->add_literal(migraphx::literal(ds, data1));
        auto a1                  = then_mod->add_instruction(migraphx::make_op("add"), x, l1);
        then_mod->add_return({a1});

        auto* else_mod           = p.create_module("If_0_else");
        std::vector<float> data2 = {-0.258047, 0.360394, 0.536804, -0.577762, 1.0217, 1.02442};
        auto l2                  = else_mod->add_literal(migraphx::literal(ds, data2));
        auto a2                  = else_mod->add_instruction(migraphx::make_op("mul"), y, l2);
        else_mod->add_return({a2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({r});

        return p;
    }
};
