
#include <onnx_test.hpp>


TEST_CASE(unique_unsorted_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s_x{migraphx::shape::float_type, {6}};
    std::vector<float> x_data = {2, 1, 1, 3, 4, 3};
    auto x                    = mm->add_literal(migraphx::literal(s_x, x_data));

    auto out   = mm->add_instruction(migraphx::make_op("unique", {{"sorted", 0}, {"axis", 0}}), x);
    auto y     = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto y_idx = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    auto x_idx = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), out);
    auto count = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 3}}), out);
    mm->add_return({y, y_idx, x_idx, count});
    auto prog = migraphx::parse_onnx("unique_unsorted_test.onnx");

    EXPECT(p == prog);
}


