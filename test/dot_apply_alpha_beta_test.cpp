#include "migraphx/instruction.hpp"
#include <migraphx/common.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

TEST_CASE(dot_apply_alpha_beta_half)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto y = m1.add_parameter("y", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto z = m1.add_parameter("z", migraphx::shape{migraphx::shape::half_type, {2, 2}});
        auto dot_res =
            migraphx::insert_dot_apply_alpha_beta(m1, m1.end(), {x, y, z}, "dot", 3.0f, 2.0f);
        m1.add_instruction(migraphx::make_op("identity"), dot_res);
    }
    migraphx::module m2;
    {

        auto ht              = migraphx::shape::half_type;
        auto ft              = migraphx::shape::float_type;
        auto x               = m2.add_parameter("x", migraphx::shape{ht, {2, 2}});
        auto y               = m2.add_parameter("y", migraphx::shape{ht, {2, 2}});
        auto z               = m2.add_parameter("z", migraphx::shape{ht, {2, 2}});
        auto alpha_literal   = m2.add_literal(3.0f);
        auto alpha_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}),
            alpha_literal);
        auto x_float = m2.add_instruction(migraphx::make_op("convert", {{"target_type", ft}}), x);
        auto x_alpha_float = m2.add_instruction(migraphx::make_op("mul"), alpha_broadcast, x_float);
        auto x_half =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", ht}}), x_alpha_float);
        auto dot_res      = m2.add_instruction(migraphx::make_op("dot"), x_half, y);
        auto beta_literal = m2.add_literal(2.0f);
        auto z_float = m2.add_instruction(migraphx::make_op("convert", {{"target_type", ft}}), z);
        auto beta_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", z->get_shape().lens()}}),
            beta_literal);
        auto z_beta_float = m2.add_instruction(migraphx::make_op("mul"), z_float, beta_broadcast);
        auto z_beta_half =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", ht}}), z_beta_float);
        auto z_add = m2.add_instruction(migraphx::make_op("add"), dot_res, z_beta_half);
        m2.add_instruction(migraphx::make_op("identity"), z_add);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(dot_apply_alpha_beta_double)
{
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto y       = m1.add_parameter("y", migraphx::shape{migraphx::shape::double_type, {2, 2}});
        auto z       = m1.add_parameter("z", migraphx::shape{migraphx::shape::double_type, {2, 1}});
        auto dot_res = migraphx::add_dot_apply_alpha_beta<float>(m1, {x, y, z}, "dot", 3, 2);
        m1.add_instruction(migraphx::make_op("identity"), dot_res);
    }
    migraphx::module m2;
    {

        auto dt              = migraphx::shape::double_type;
        auto x               = m2.add_parameter("x", migraphx::shape{dt, {2, 2}});
        auto y               = m2.add_parameter("y", migraphx::shape{dt, {2, 2}});
        auto z               = m2.add_parameter("z", migraphx::shape{dt, {2, 1}});
        auto alpha_literal   = m2.add_literal(3.0f);
        auto alpha_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}),
            alpha_literal);
        auto alpha_double = m2.add_instruction(migraphx::make_op("convert", {{"target_type", dt}}),
                                               alpha_broadcast);
        auto x_alpha_double = m2.add_instruction(migraphx::make_op("mul"), alpha_double, x);
        auto dot_res        = m2.add_instruction(migraphx::make_op("dot"), x_alpha_double, y);
        auto z_broadcast =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}), z);
        auto beta_literal   = m2.add_literal(2.0f);
        auto beta_broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", z_broadcast->get_shape().lens()}}),
            beta_literal);
        auto beta_double =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", dt}}), beta_broadcast);
        auto z_beta_double = m2.add_instruction(migraphx::make_op("mul"), z_broadcast, beta_double);
        auto z_add         = m2.add_instruction(migraphx::make_op("add"), dot_res, z_beta_double);
        m2.add_instruction(migraphx::make_op("identity"), z_add);
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
