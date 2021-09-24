#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::program& p) { migraphx::run_passes(p, {migraphx::fuse_pointwise{}}); }

template<class F>
migraphx::instruction_ref add_pointwise(migraphx::program& p, std::string name, std::vector<migraphx::instruction_ref> inputs, F f)
{
    auto* pm = p.create_module(name);
    auto* mm = p.get_main_module();
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(params), [&](auto input) {
        return pm->add_parameter("x" + std::to_string(params.size()), migraphx::shape{input->get_shape().type()});
    });
    f(pm, params);
    return mm->add_instruction(migraphx::make_op("pointwise"), inputs, {pm});
}

auto single_pointwise(std::string name)
{
    return [=](auto* pm, const auto& inputs) { pm->add_instruction(migraphx::make_op(name), inputs); };
}

TEST_CASE(simple)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", s);
        auto z = mm->add_parameter("z", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), pass, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", s);
        auto z = mm->add_parameter("z", s);
        auto add1 = add_pointwise(p2, "pointwise0", {x, y}, single_pointwise("add"));
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = add_pointwise(p2, "pointwise1", {pass, z}, single_pointwise("add"));
        mm->add_return({add2});
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
