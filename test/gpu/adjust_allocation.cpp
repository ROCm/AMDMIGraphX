#include <migraphx/gpu/adjust_allocation.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/contiguous.hpp>
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
        return {migraphx::auto_contiguous{},
                migraphx::gpu::lowering{ctx},
                migraphx::dead_code_elimination{},
                migraphx::eliminate_contiguous{},
                migraphx::dead_code_elimination{}};
    }
    migraphx::gpu::context get_context() const { return migraphx::gpu::context{}; }
};

TEST_CASE(tanh_shape)
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto x   = p.add_parameter("x", s);
        auto tx  = p.add_instruction(migraphx::op::transpose{{1, 0}}, x);
        auto txh = p.add_instruction(migraphx::op::tanh{}, tx);
        auto sum = p.add_instruction(migraphx::op::add{}, txh, txh);
        p.add_instruction(migraphx::op::contiguous{}, sum);

        return p;
    };

    auto p1 = create_program();
    auto p2 = create_program();
    EXPECT(p1 == p2);

    p1.compile(lowering_target{});
    p2.compile(lowering_target());

    EXPECT(p1 == p2);

    for(auto ins : iterator_for(p1))
    {
        if(ins->name() == "hip::allocate")
        {
            migraphx::shape new_s{migraphx::shape::float_type, {3, 2}, {1, 3}};
            ins->replace(migraphx::gpu::hip_allocate{new_s});
        }
    }
    EXPECT(p1 != p2);

    migraphx::run_passes(p2,
                         {migraphx::gpu::adjust_allocation{}, migraphx::dead_code_elimination{}});
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
