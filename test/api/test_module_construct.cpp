#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(add_op)
{
    migraphx::program p;
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    auto y             = m.add_parameter("y", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    auto add_op        = migraphx::operation("add");
    m.add_instruction(add_op, {x, y});
    m.print();
}

TEST_CASE(if_then_else_op)
{
    migraphx::program p;
    auto mm = p.get_main_module();
    migraphx::shape cond_s{migraphx_shape_bool_type};
    auto cond = mm.add_parameter("cond", cond_s);

    auto then_mod = p.create_module("If_0_if");
    auto z        = then_mod.add_parameter("x", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    then_mod.add_return({z});

    auto else_mod = p.create_module("If_0_else");
    z             = else_mod.add_parameter("y", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    else_mod.add_return({z});

    auto ret          = mm.add_instruction(migraphx::operation("if"), {cond}, {then_mod, else_mod});
    auto get_tuple_op = migraphx::operation("get_tuple_elem", "{index: 0}");
    mm.add_instruction(get_tuple_op, {ret});
    mm.print();
    else_mod.print();
    then_mod.print();
}

TEST_CASE(instructions_refs) {
    migraphx::program p;
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    auto y             = m.add_parameter("y", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    migraphx::api::instructions_refs args{x, y};
    EXPECT(args.size() == 2);
    EXPECT(migraphx::as_address(args[0]) == migraphx::as_address(x));
    EXPECT(migraphx::as_address(args[1]) == migraphx::as_address(y));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
