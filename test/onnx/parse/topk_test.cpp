
#include <onnx_test.hpp>


TEST_CASE(topk_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sk{migraphx::shape::int64_type, {1}};
    mm->add_literal(migraphx::literal(sk, {4}));
    migraphx::shape s{migraphx::shape::float_type, {2, 5, 3, 2}};
    auto data = mm->add_parameter("data", s);
    auto out  = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 4}, {"axis", 1}, {"largest", 0}}), data);
    auto val = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    mm->add_return({val, ind});

    auto prog = migraphx::parse_onnx("topk_test.onnx");

    EXPECT(p == prog);
}


