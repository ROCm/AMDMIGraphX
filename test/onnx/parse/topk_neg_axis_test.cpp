
#include <onnx_test.hpp>

TEST_CASE(topk_neg_axis_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sk{migraphx::shape::int64_type, {1}};
    mm->add_literal(migraphx::literal(sk, {3}));
    migraphx::shape s{migraphx::shape::float_type, {3, 4, 5, 6}};
    auto data = mm->add_parameter("data", s);
    auto out  = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 3}, {"axis", -2}, {"largest", 1}}), data);
    auto val = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    mm->add_return({val, ind});

    auto prog = migraphx::parse_onnx("topk_neg_axis_test.onnx");

    EXPECT(p == prog);
}
