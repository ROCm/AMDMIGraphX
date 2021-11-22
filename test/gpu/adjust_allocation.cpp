#include <migraphx/gpu/allocation_model.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/adjust_allocation.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/contiguous.hpp>
#include <migraphx/op/tanh.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_lowering(migraphx::program& p, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(*p.get_main_module(),
                         {migraphx::auto_contiguous{},
                          migraphx::gpu::lowering{&ctx, offload_copy},
                          migraphx::dead_code_elimination{},
                          migraphx::eliminate_contiguous{"gpu::contiguous"},
                          migraphx::dead_code_elimination{}});
}

TEST_CASE(tanh_shape)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto x   = mm->add_parameter("x", s);
        auto tx  = mm->add_instruction(migraphx::op::transpose{{1, 0}}, x);
        auto txh = mm->add_instruction(migraphx::op::tanh{}, tx);
        auto sum = mm->add_instruction(migraphx::op::add{}, txh, txh);
        mm->add_instruction(migraphx::op::contiguous{}, sum);

        return p;
    };

    auto p1 = create_program();
    auto p2 = create_program();
    EXPECT(p1 == p2);

    run_lowering(p1);
    run_lowering(p2);

    EXPECT(p1 == p2);

    for(auto ins : iterator_for(*p1.get_main_module()))
    {
        if(ins->name() == "hip::allocate")
        {
            migraphx::shape new_s{migraphx::shape::float_type, {3, 2}, {1, 3}};
            ins->replace(migraphx::gpu::hip_allocate{new_s});
        }
    }
    EXPECT(p1 != p2);

    migraphx::run_passes(*p2.get_main_module(),
                         {migraphx::adjust_allocation{migraphx::gpu::gpu_allocation_model{}},
                          migraphx::dead_code_elimination{}});
    EXPECT(p1 == p2);
}

TEST_CASE(no_copy_dead_param)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto x = mm->add_parameter("x", s);
        mm->add_parameter("y", s);
        auto sum = mm->add_instruction(migraphx::make_op("add"), x, x);
        mm->add_return({sum});

        return p;
    };

    auto create_gpu_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto x = mm->add_parameter("x", s);
        mm->add_parameter("y", s);
        auto xb = mm->add_instruction(migraphx::make_op("hip::allocate", {{"shape", to_value(s)}}));
        auto gx = mm->add_instruction(migraphx::make_op("hip::copy_to_gpu"), x, xb);
        auto ab = mm->add_instruction(migraphx::make_op("hip::allocate", {{"shape", to_value(s)}}));
        auto sum = mm->add_instruction(migraphx::make_op("gpu::add"), gx, gx, ab);
        auto r   = mm->add_instruction(migraphx::make_op("hip::copy_from_gpu"), sum);
        mm->add_return({r});

        return p;
    };

    auto p1 = create_program();
    auto p2 = create_gpu_program();

    run_lowering(p1, true);
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
