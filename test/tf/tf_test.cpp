#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/tf.hpp>
#include "test.hpp"

TEST_CASE(add_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 3}});
    p.add_instruction(migraphx::op::add{}, l0, l1);
    auto prog = migraphx::parse_tf("add_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(add_bcast_test)
{

    migraphx::program p;
    migraphx::shape s0{migraphx::shape::float_type, {2, 3}};
    auto l0 = p.add_parameter("0", s0);
    auto l1 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 1}});
    auto l2 = p.add_instruction(migraphx::op::multibroadcast{s0.lens()}, l0);
    auto l3 = p.add_instruction(migraphx::op::multibroadcast{s0.lens()}, l1);
    p.add_instruction(migraphx::op::add{}, l2, l3);
    auto prog = migraphx::parse_tf("add_bcast_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(biasadd_test)
{
    migraphx::program p;
    migraphx::shape s0{migraphx::shape::float_type, {1, 1, 1, 500}};
    uint64_t axis = 1;
    auto l0       = p.add_parameter("0", s0);
    auto l1       = p.add_instruction(migraphx::op::transpose{{0, 3, 1, 2}}, l0);
    auto l2       = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {500}});
    auto l3       = p.add_instruction(migraphx::op::broadcast{axis, l1->get_shape()}, l2);
    p.add_instruction(migraphx::op::add{}, l1, l3);
    auto prog = migraphx::parse_tf("biasadd_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(identity_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::identity{}, l0);
    auto prog = migraphx::parse_tf("identity_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(pooling_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 16, 16, 3}});
    migraphx::op::pooling avg_pool_op{"average"};
    migraphx::op::pooling max_pool_op{"max"};
    avg_pool_op.padding_mode = migraphx::op::padding_mode_t::valid;
    max_pool_op.padding_mode = migraphx::op::padding_mode_t::valid;
    avg_pool_op.stride       = {2, 2};
    max_pool_op.stride       = {2, 2};
    avg_pool_op.lengths      = {2, 2};
    max_pool_op.lengths      = {2, 2};
    auto l1                  = p.add_instruction(migraphx::op::transpose{{0, 3, 1, 2}}, l0);
    p.add_instruction(max_pool_op, l1);
    auto l2 = p.add_instruction(migraphx::op::transpose{{0, 3, 1, 2}}, l0);
    p.add_instruction(avg_pool_op, l2);
    auto prog = migraphx::parse_tf("pooling_test.pb", true);

    EXPECT(p == prog);
}

TEST_CASE(relu_test)
{
    migraphx::program p;
    auto l0 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    p.add_instruction(migraphx::op::relu{}, l0);
    auto prog = migraphx::parse_tf("relu_test.pb", false);

    EXPECT(p == prog);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
