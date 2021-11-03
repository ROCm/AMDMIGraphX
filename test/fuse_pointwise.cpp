#include "migraphx/dead_code_elimination.hpp"
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::fuse_pointwise{}, migraphx::dead_code_elimination{}});
}

template <class F>
migraphx::instruction_ref add_pointwise(migraphx::program& p,
                                        const std::string& name,
                                        std::vector<migraphx::instruction_ref> inputs,
                                        F f)
{
    auto* pm = p.create_module(name);
    auto* mm = p.get_main_module();
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(params), [&](auto input) {
        return pm->add_parameter("x" + std::to_string(params.size()),
                                 migraphx::shape{input->get_shape().type()});
    });
    auto r = f(pm, params);
    pm->add_return({r});
    return mm->add_instruction(migraphx::make_op("pointwise"), inputs, {pm});
}

auto single_pointwise(const std::string& name)
{
    return [=](auto* pm, const auto& inputs) {
        return pm->add_instruction(migraphx::make_op(name), inputs);
    };
}

TEST_CASE(single)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), pass, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = add_pointwise(p2, "pointwise0", {x, y}, single_pointwise("add"));
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = add_pointwise(p2, "pointwise1", {pass, z}, single_pointwise("add"));
        mm->add_return({add2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(double_add)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto fadd = add_pointwise(p2, "pointwise0", {x, y, z}, [=](auto* pm, const auto& inputs) {
            auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
            return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
        });
        mm->add_return({fadd});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(used_twice_not_fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, y);
        auto add3 = mm->add_instruction(migraphx::make_op("add"), pass, add2);
        mm->add_return({add3});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = add_pointwise(p2, "pointwise0", {x, y}, single_pointwise("add"));
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto fadd =
            add_pointwise(p2, "pointwise1", {add1, y, pass}, [=](auto* pm, const auto& inputs) {
                auto add2 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), inputs[2], add2);
            });
        mm->add_return({fadd});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(used_twice_fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, x);
        auto add3 = mm->add_instruction(migraphx::make_op("add"), add1, y);
        auto add4 = mm->add_instruction(migraphx::make_op("add"), add2, add3);
        mm->add_return({add4});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto fadd = add_pointwise(p2, "pointwise0", {x, y}, [=](auto* pm, const auto& inputs) {
            auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
            auto add2 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[0]);
            auto add3 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[1]);
            return pm->add_instruction(migraphx::make_op("add"), add2, add3);
        });
        mm->add_return({fadd});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(duplicate_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, x);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), pass, y);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = add_pointwise(p2, "pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[0]);
        });
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = add_pointwise(p2, "pointwise1", {pass, y}, single_pointwise("add"));
        mm->add_return({add2});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(scalar_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto one = mm->add_literal(1.0f);
        auto y =
            mm->add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", s.lens()}}), one);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({add1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add1 = add_pointwise(p2, "pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            auto y = pm->add_literal(1.0f);
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], y);
        });
        mm->add_return({add1});
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
