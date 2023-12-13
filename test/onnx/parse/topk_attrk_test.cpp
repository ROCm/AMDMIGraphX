
#include <onnx_test.hpp>


TEST_CASE(topk_attrk_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 5, 3, 2}};
    auto data = mm->add_parameter("data", s);
    auto out  = mm->add_instruction(migraphx::make_op("topk", {{"k", 2}, {"axis", -1}}), data);
    auto val  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    mm->add_return({val, ind});

    auto prog = migraphx::parse_onnx("topk_attrk_test.onnx");

    EXPECT(p == prog);
}


