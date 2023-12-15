
#include <onnx_test.hpp>


TEST_CASE(slice_var_input_dyn1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto data =
        mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {{3, 8}, {2, 2}}});
    auto starts = mm->add_parameter("starts", migraphx::shape{migraphx::shape::int32_type, {2}});
    auto ends   = mm->add_parameter("ends", migraphx::shape{migraphx::shape::int32_type, {2}});
    auto axes   = mm->add_parameter("axes", migraphx::shape{migraphx::shape::int32_type, {2}});
    auto ret    = mm->add_instruction(migraphx::make_op("slice"), data, starts, ends, axes);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {3, 8};
    auto prog                     = parse_onnx("slice_var_input_dyn1.onnx", options);
    EXPECT(p == prog);
}


