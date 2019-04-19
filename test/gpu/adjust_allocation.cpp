#include <migraphx/gpu/adjust_allocation.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/tanh.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct lowering_target
{
    std::string name() const { return "gpu::lowering"; }
    std::vector<migraphx::pass> get_passes(migraphx::context& gctx) const
    {
        auto& ctx = migraphx::any_cast<migraphx::gpu::context>(gctx);
        return {migraphx::gpu::lowering{ctx}, migraphx::dead_code_elimination{}};
    }
    migraphx::gpu::context get_context() const { return migraphx::gpu::context{}; }
};

TEST_CASE(trans_tanh)
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto x  = p.add_parameter("x", s);
        auto sm = p.add_instruction(migraphx::op::add{}, x, x);
        p.add_instruction(migraphx::op::tanh{}, sm);

        return p;
    };

    auto p1 = create_program();
    auto p2 = create_program();
    EXPECT(p1 == p2);

    // relace the add instruction with using a incorrect
    // output shape
    for(auto ins : iterator_for(p1))
    {
        if(ins->name() == "add")
        {
            migraphx::shape wrong_s{migraphx::shape::float_type, {3, 2}};
            migraphx::instruction::replace(ins, ins->get_operator(), wrong_s, ins->inputs());
        }

        if(ins->name() == "tanh")
        {
            migraphx::shape orig_s{migraphx::shape::float_type, {2, 3}};
            migraphx::instruction::replace(ins, ins->get_operator(), orig_s, ins->inputs());
        }
    }
    EXPECT(p1 != p2);

    p1.compile(lowering_target{});
    p2.compile(lowering_target{});
    EXPECT(p1 != p2);

    for(auto ins : iterator_for(p1))
    {
        if(ins->name() == "gpu::add")
        {
            migraphx::shape correct_s{migraphx::shape::float_type, {2, 3}};
            migraphx::instruction::replace(ins, ins->get_operator(), correct_s, ins->inputs());
        }
    }
    EXPECT(p1 != p2);

    migraphx::run_passes(p1,
                         {migraphx::gpu::adjust_allocation{}, migraphx::dead_code_elimination{}});
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
