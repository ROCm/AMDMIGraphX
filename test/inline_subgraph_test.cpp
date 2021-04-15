#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/inline_subgraph.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::inline_subgraph{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(cannot_inline_both)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {2, 3}};
        auto x = mm->add_parameter("x", sd);

        std::vector<float> one(sd.elements(), 1);
        std::vector<float> two(sd.elements(), 2);

        auto* then_smod = p.create_module("then_smod");
        auto l1         = then_smod->add_literal(migraphx::literal{sd, one});
        auto r1         = then_smod->add_instruction(migraphx::make_op("add"), x, l1);
        then_smod->add_return({r1});

        auto* else_smod = p.create_module("else_smod");
        auto l2         = else_smod->add_literal(migraphx::literal{sd, two});
        auto r2         = else_smod->add_instruction(migraphx::make_op("mul"), x, l2);
        else_smod->add_return({r2});

        migraphx::shape s_cond{migraphx::shape::bool_type, {1}};
        auto cond = mm->add_parameter("cond", s_cond);
        auto ret  = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_smod, else_smod});
        mm->add_return({ret});

        return p;
    };

    auto p = create_program();
    auto* mm = p.get_main_module();
    run_pass(*mm);

    EXPECT(p == create_program());
}

TEST_CASE(cannot_inline_one)
{
    auto create_program = [] {
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
        mm->add_return({ret});

        return p;
    };

    auto p = create_program();
    auto* mm = p.get_main_module();
    run_pass(*mm);

    EXPECT(p == create_program());
}

TEST_CASE(inline_subgraph)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", cond_s);

        migraphx::shape s{migraphx::shape::float_type, {5}};

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {1, 2, 3, 4, 5};
        auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
        then_mod->add_return({l1});

        auto* else_mod           = p.create_module("If_0_else");
        std::vector<float> data2 = {5, 4, 3, 2, 1};
        auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
        else_mod->add_return({l2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({ret});

        return p;
    };

    auto p = create_program();
    auto* mm = p.get_main_module();
    run_pass(*mm);

    auto create_inlined = [] {
        migraphx::shape s{migraphx::shape::float_type, {5}};
        std::vector<float> data1 = {1, 2, 3, 4, 5};
        std::vector<float> data2 = {5, 4, 3, 2, 1};
        migraphx::shape cond_s{migraphx::shape::bool_type};

        migraphx::program pi;
        auto* mm = pi.get_main_module();
        auto cond = mm->add_parameter("cond", cond_s);
        auto l1   = mm->add_literal(migraphx::literal(s, data1));
        auto l2   = mm->add_literal(migraphx::literal(s, data2));
        auto icond = mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), cond);
        auto mcond = mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {5}}}), icond);
        auto ccond = mm->add_instruction(migraphx::make_op("contiguous"), mcond);
        auto cl = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), l1, l2);
        auto rl = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {10}}}), cl);
        auto r = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rl, ccond);
        mm->add_return({r});

        return pi;
    };

    EXPECT(p == create_inlined());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
