#include <migraphx/propagate_precision.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/common.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::propagate_precision{}, migraphx::eliminate_common_subexpression{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(propagate_input)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto y        = m1.add_parameter("y", s2);
        auto two      = m1.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div      = migraphx::add_common_op(m1, migraphx::make_op("div"), {x, two});
        auto sqrt     = m1.add_instruction(migraphx::make_op("sqrt"), div);
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), sqrt);
        auto mul      = m1.add_instruction(migraphx::make_op("mul"), convert1, y);
        auto convert2 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), mul);
        m1.add_return({convert2});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto y        = m2.add_parameter("y", s2);
        auto convert1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto two      = m2.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div      = migraphx::add_common_op(m2, migraphx::make_op("div"), {convert1, two});
        auto sqrt     = m2.add_instruction(migraphx::make_op("sqrt"), div);
        auto mul      = m2.add_instruction(migraphx::make_op("mul"), sqrt, y);
        auto convert2 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), mul);
        m2.add_return({convert2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(propagate_output)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {2, 3}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto y        = m1.add_parameter("y", s2);
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto two  = m1.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div  = migraphx::add_common_op(m1, migraphx::make_op("div"), {convert1, two});
        auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), div);
        auto mul  = m1.add_instruction(migraphx::make_op("mul"), sqrt, y);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto y        = m2.add_parameter("y", s2);
        auto two      = m2.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div      = migraphx::add_common_op(m2, migraphx::make_op("div"), {x, two});
        auto sqrt     = m2.add_instruction(migraphx::make_op("sqrt"), div);
        auto convert1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), sqrt);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), convert1, y);
        m2.add_return({mul});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(propagate_conflict)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::double_type, {2, 3}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto y        = m1.add_parameter("y", s2);
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto two  = m1.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div  = migraphx::add_common_op(m1, migraphx::make_op("div"), {convert1, two});
        auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), div);
        auto convert2 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::double_type}}), sqrt);
        auto mul  = m1.add_instruction(migraphx::make_op("mul"), convert2, y);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto y        = m2.add_parameter("y", s2);
        auto convert1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::double_type}}), x);
        auto two  = m2.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div  = migraphx::add_common_op(m2, migraphx::make_op("div"), {convert1, two});
        auto sqrt = m2.add_instruction(migraphx::make_op("sqrt"), div);
        auto mul  = m2.add_instruction(migraphx::make_op("mul"), sqrt, y);
        m2.add_return({mul});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(propagate_reduce)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 3}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto three  = m1.add_literal(migraphx::literal{{migraphx::shape::half_type}, {3}});
        auto squared = m1.add_instruction(migraphx::make_op("mul"), x, x);
        auto div = migraphx::add_common_op(m1, migraphx::make_op("div"), {squared, three});
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), div);
        auto reduce = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", 1}}), convert1);
        auto convert2 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), reduce);
        auto sqrt     = m1.add_instruction(migraphx::make_op("sqrt"), convert2);
        auto mul      = migraphx::add_common_op(m1, migraphx::make_op("mul"), {x, sqrt});
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto convert1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto three  = m2.add_literal(migraphx::literal{{migraphx::shape::half_type}, {3}});
        auto squared = m2.add_instruction(migraphx::make_op("mul"), convert1, convert1);
        auto div = migraphx::add_common_op(m2, migraphx::make_op("div"), {squared, three});
        auto reduce = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", 1}}), div);
        auto sqrt     = m2.add_instruction(migraphx::make_op("sqrt"), reduce);
        auto convert2 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), sqrt);
        auto mul      = migraphx::add_common_op(m2, migraphx::make_op("mul"), {x, convert2});
        m2.add_return({mul});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
