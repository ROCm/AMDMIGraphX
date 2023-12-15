
#include <onnx_test.hpp>

TEST_CASE(loop_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape si{migraphx::shape::int64_type, {1}};
    auto max_iter = mm->add_parameter("max_trip_count", si);
    migraphx::shape sc{migraphx::shape::bool_type, {1}};
    auto icond = mm->add_parameter("keep_going_cond", sc);
    migraphx::shape su{migraphx::shape::float_type, {1}};
    auto a = mm->add_parameter("a", su);
    auto b = mm->add_parameter("b", su);

    auto* body = p.create_module("Loop_4_loop");
    body->add_parameter("iteration_num", si);
    body->add_parameter("keep_going_inp", sc);
    auto var = body->add_parameter("b_in", su);

    auto ad = body->add_instruction(migraphx::make_op("add"), a, var);
    auto sb = body->add_instruction(migraphx::make_op("sub"), a, var);
    auto gt = body->add_instruction(migraphx::make_op("greater"), ad, sb);
    auto cv = body->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
    auto ad1 = body->add_instruction(migraphx::make_op("add"), sb, sb);
    body->add_return({cv, sb, ad, ad1});

    auto lp = mm->add_instruction(
        migraphx::make_op("loop", {{"max_iterations", 10}}), {max_iter, icond, b}, {body});
    auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), lp);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), lp);
    auto r2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), lp);
    mm->add_return({r0, r2});

    auto prog = migraphx::parse_onnx("loop_test.onnx");

    EXPECT(p == prog);
}
