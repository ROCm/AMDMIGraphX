
#include <onnx_test.hpp>


TEST_CASE(slice_var_input_static0)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto data   = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 2}});
    auto starts = mm->add_parameter("starts", migraphx::shape{migraphx::shape::int32_type, {2}});
    auto ends   = mm->add_parameter("ends", migraphx::shape{migraphx::shape::int32_type, {2}});
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {0, 1}}}), data, starts, ends);
    auto prog = optimize_onnx("slice_var_input_static0.onnx");

    EXPECT(p == prog);
}


