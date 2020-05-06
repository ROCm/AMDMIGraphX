#include <migraphx/decompose.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/op/abnormal_ops.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/op/multibroadcast.hpp>
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

TEST_CASE(dot_add_beta_float)
{
    migraphx::program p1;
    {
        auto x   = p1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto y   = p1.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto z   = p1.add_parameter("z", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto dot = p1.add_instruction(migraphx::op::dot{1.0, 0.5}, x, y, z);
        p1.add_instruction(migraphx::op::identity{}, dot);
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto x   = p2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto y   = p2.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto z   = p2.add_parameter("z", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto dot = p2.add_instruction(migraphx::op::dot{1, 0}, x, y);
        auto beta =
            p2.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {0.5}});
        auto beta_broadcast = p2.add_instruction(migraphx::op::multibroadcast{{2, 2}}, beta);
        auto mul            = p2.add_instruction(migraphx::op::mul{}, z, beta_broadcast);
        auto add            = p2.add_instruction(migraphx::op::add{}, dot, mul);
        p2.add_instruction(migraphx::op::identity{}, add);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(dot_add_beta_half)
{
    migraphx::program p1;
    {
        auto x   = p1.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto y   = p1.add_parameter("y", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto z   = p1.add_parameter("z", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto dot = p1.add_instruction(migraphx::op::dot{1.0, 0.5}, x, y, z);
        p1.add_instruction(migraphx::op::identity{}, dot);
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto x   = p2.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto y   = p2.add_parameter("y", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto z   = p2.add_parameter("z", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto dot = p2.add_instruction(migraphx::op::dot{1, 0}, x, y);
        auto beta =
            p2.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {0.5}});
        auto beta_broadcast = p2.add_instruction(migraphx::op::multibroadcast{{2, 2}}, beta);
        auto mul            = p2.add_instruction(migraphx::op::mul{}, z, beta_broadcast);
        auto add            = p2.add_instruction(migraphx::op::add{}, dot, mul);
        p2.add_instruction(migraphx::op::identity{}, add);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(dot_add_beta_double)
{
    migraphx::program p1;
    {
        auto x   = p1.add_parameter("x", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto y   = p1.add_parameter("y", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto z   = p1.add_parameter("z", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto dot = p1.add_instruction(migraphx::op::dot{1.0, 0.5}, x, y, z);
        p1.add_instruction(migraphx::op::identity{}, dot);
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto x   = p2.add_parameter("x", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto y   = p2.add_parameter("y", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto z   = p2.add_parameter("z", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto dot = p2.add_instruction(migraphx::op::dot{1, 0}, x, y);
        auto beta =
            p2.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::double_type}, {0.5}});
        auto beta_broadcast = p2.add_instruction(migraphx::op::multibroadcast{{2, 2}}, beta);
        auto mul            = p2.add_instruction(migraphx::op::mul{}, z, beta_broadcast);
        auto add            = p2.add_instruction(migraphx::op::add{}, dot, mul);
        p2.add_instruction(migraphx::op::identity{}, add);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(dot_add_beta_int)
{
    migraphx::program p1;
    {
        auto x   = p1.add_parameter("x", migraphx::shape{migraphx::shape::int32_type, {2, 2}});
        auto y   = p1.add_parameter("y", migraphx::shape{migraphx::shape::int32_type, {2, 2}});
        auto z   = p1.add_parameter("z", migraphx::shape{migraphx::shape::int32_type, {2, 2}});
        auto dot = p1.add_instruction(migraphx::op::dot{1.0, 0.5}, x, y, z);
        p1.add_instruction(migraphx::op::identity{}, dot);
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
