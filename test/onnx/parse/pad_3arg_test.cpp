
#include <onnx_test.hpp>


TEST_CASE(pad_3arg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    mm->add_literal({migraphx::shape{migraphx::shape::float_type}, {1.0f}});
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 1, 2, 2}});
    auto r = mm->add_instruction(
        migraphx::make_op("pad", {{"pads", {1, 1, 2, 2}}, {"value", 1.0f}}), l0);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("pad_3arg_test.onnx");

    EXPECT(p == prog);
}


