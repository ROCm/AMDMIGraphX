
#include <onnx_test.hpp>

TEST_CASE(pad_4arg_neg_axes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 5}});
    // axes=[-3,-1]
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {2}}, {-3, -1}});
    // constant_value=1
    mm->add_literal({migraphx::shape{migraphx::shape::float_type}, {1.0f}});
    // pads=[1,3,2,4]
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 3, 2, 4}});
    auto r = mm->add_instruction(
        migraphx::make_op("pad", {{"pads", {0, 1, 0, 3, 0, 2, 0, 4}}, {"value", 1.0f}}), l0);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("pad_4arg_neg_axes_test.onnx");

    EXPECT(p == prog);
}
