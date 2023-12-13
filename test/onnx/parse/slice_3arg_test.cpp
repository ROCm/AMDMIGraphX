
#include <onnx_test.hpp>

TEST_CASE(slice_3arg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 5}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {0, 0}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {2, 5}});
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 0}}, {"ends", {2, 5}}}), l0);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("slice_3arg_test.onnx");

    EXPECT(p == prog);
}
