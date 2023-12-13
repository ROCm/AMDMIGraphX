
#include <onnx_test.hpp>


TEST_CASE(slice_5arg_step_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 5}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-2, 2}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-1, -2}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-5, -1}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-1, -3}});
    auto slice_out = mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {-1, -2}}, {"starts", {-4, -3}}, {"ends", {2147483647, -1}}}),
        l0);
    auto reverse_out =
        mm->add_instruction(migraphx::make_op("reverse", {{"axes", {-1}}}), slice_out);
    auto step_out = mm->add_instruction(
        migraphx::make_op("step", {{"axes", {-1, -2}}, {"steps", {2, 2}}}), reverse_out);
    mm->add_return({step_out});

    auto prog = migraphx::parse_onnx("slice_5arg_step_test.onnx");

    EXPECT(p == prog);
}


