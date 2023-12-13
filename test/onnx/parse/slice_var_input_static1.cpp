
#include <onnx_test.hpp>


TEST_CASE(slice_var_input_static1)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto data   = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 2}});
    auto starts = mm->add_parameter("starts", migraphx::shape{migraphx::shape::int64_type, {2}});
    auto ends   = mm->add_parameter("ends", migraphx::shape{migraphx::shape::int64_type, {2}});
    auto axes   = mm->add_parameter("axes", migraphx::shape{migraphx::shape::int64_type, {2}});
    mm->add_instruction(migraphx::make_op("slice"), data, starts, ends, axes);
    auto prog = optimize_onnx("slice_var_input_static1.onnx");

    EXPECT(p == prog);
}


