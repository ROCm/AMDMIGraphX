#include <test.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_ops.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::fuse_ops{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(test_add_sigmoid)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s   = migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}};
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto add_buffer =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto add = mm->add_instruction(migraphx::make_op("gpu::add"), x, y, add_buffer);
        auto sigmoid_buffer =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        mm->add_instruction(migraphx::make_op("gpu::sigmoid"), add, sigmoid_buffer);
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s   = migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}};
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto buffer =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        mm->add_instruction(migraphx::make_op("gpu::add_sigmoid"), {x, y, buffer});
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
