
#include <onnx_test.hpp>


TEST_CASE(squeeze_axes_input_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal({migraphx::shape::int64_type, {2}}, {1, 3}));
    auto l0 = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 1, 5, 1}});
    auto l1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 3}}}), l0);
    mm->add_return({l1});

    auto prog = migraphx::parse_onnx("squeeze_axes_input_test.onnx");

    EXPECT(p == prog);
}


