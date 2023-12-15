
#include <onnx_test.hpp>


TEST_CASE(unique_dynamic_sorted_3D_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::int64_type, {4, 4, 4}};
    auto x = mm->add_parameter("X", s);

    auto out   = mm->add_instruction(migraphx::make_op("unique", {{"sorted", 1}}), x);
    auto y     = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto y_ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    auto x_ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), out);
    auto count = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 3}}), out);

    mm->add_return({y, y_ind, x_ind, count});
    auto prog = migraphx::parse_onnx("unique_dynamic_sorted_3D_test.onnx");

    EXPECT(p == prog);
}


