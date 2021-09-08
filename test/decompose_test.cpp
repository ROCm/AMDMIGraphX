#include <migraphx/decompose.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m) { migraphx::run_passes(m, {migraphx::decompose{}}); }

TEST_CASE(quant_dot_add)
{
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto y     = m1.add_parameter("y", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto z     = m1.add_parameter("z", migraphx::shape{migraphx::shape::int32_type, {2, 2}});
        auto q_dot = m1.add_instruction(migraphx::make_op("quant_dot"), x, y, z);
        m1.add_instruction(migraphx::make_op("identity"), q_dot);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto y = m2.add_parameter("y", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto z = m2.add_parameter("z", migraphx::shape{migraphx::shape::int32_type, {2, 2}});
        auto q_dot =
            m2.add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), x, y);
        auto add = m2.add_instruction(migraphx::make_op("add"), q_dot, z);
        m2.add_instruction(migraphx::make_op("identity"), add);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(quant_dot_add_beta)
{
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto y     = m1.add_parameter("y", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto z     = m1.add_parameter("z", migraphx::shape{migraphx::shape::int32_type, {2, 2}});
        auto q_dot = m1.add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1.0}, {"beta", 2}}), x, y, z);
        m1.add_instruction(migraphx::make_op("identity"), q_dot);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto y = m2.add_parameter("y", migraphx::shape{migraphx::shape::int8_type, {2, 2}});
        auto z = m2.add_parameter("z", migraphx::shape{migraphx::shape::int32_type, {2, 2}});
        auto q_dot =
            m2.add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), x, y);
        auto beta =
            m2.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {2}});
        auto beta_broadcast =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}), beta);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), z, beta_broadcast);
        auto add = m2.add_instruction(migraphx::make_op("add"), q_dot, mul);
        m2.add_instruction(migraphx::make_op("identity"), add);
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
