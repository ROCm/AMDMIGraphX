#include <migraphx/decompose.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/op/abnormal_ops.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/op/dot.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p) { migraphx::run_passes(p, {migraphx::decompose{}}); }

TEST_CASE(dot_add)
{
    migraphx::program p1;
    {
        auto x   = p1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto y   = p1.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto z   = p1.add_parameter("z", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto dot = p1.add_instruction(migraphx::op::dot{}, x, y, z);
        p1.add_instruction(migraphx::op::identity{}, dot);
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto x   = p2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto y   = p2.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto z   = p2.add_parameter("z", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto dot = p2.add_instruction(migraphx::op::dot{1, 0}, x, y);
        auto add = p2.add_instruction(migraphx::op::add{}, dot, z);
        p2.add_instruction(migraphx::op::identity{}, add);
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
