#include <migraphx/auto_contiguous.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p) { migraphx::run_passes(p, {migraphx::auto_contiguous{}}); }

// TODO: Add this test case
void literal_broadcast()
{
    migraphx::program p;
    p.add_literal(get_2_broadcasted());
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().broadcasted());
    run_pass(p);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().broadcasted());
}

TEST_CASE(literal_transpose)
{
    migraphx::program p;
    p.add_literal(get_2x2_transposed());
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    run_pass(p);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
}

TEST_CASE(after_literal_transpose)
{
    migraphx::program p;
    auto l = p.add_literal(get_2x2());
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    auto t = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    p.add_instruction(pass_op{}, t);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    run_pass(p);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
}

TEST_CASE(after_literal_broadcast)
{
    migraphx::program p;
    auto l1 = p.add_literal(get_2x2());
    auto l2 = p.add_literal(get_2());
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().broadcasted());
    auto b = p.add_instruction(migraphx::op::broadcast{0, l1->get_shape().lens()}, l2);
    p.add_instruction(pass_op{}, b);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().broadcasted());
    run_pass(p);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().broadcasted());
}

TEST_CASE(after_param_transpose)
{
    migraphx::program p;
    auto l = p.add_parameter("2x2", {migraphx::shape::float_type, {2, 2}});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    auto t = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    p.add_instruction(pass_op{}, t);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    run_pass(p);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
}

TEST_CASE(after_param_broadcast)
{
    migraphx::program p;
    auto l1 = p.add_parameter("2x2", {migraphx::shape::float_type, {2, 2}});
    auto l2 = p.add_parameter("2", {migraphx::shape::float_type, {2}});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().broadcasted());
    auto b = p.add_instruction(migraphx::op::broadcast{0, l1->get_shape().lens()}, l2);
    p.add_instruction(pass_op{}, b);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().broadcasted());
    run_pass(p);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().broadcasted());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
