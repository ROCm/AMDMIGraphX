
#include <onnx_test.hpp>

TEST_CASE(slice_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{3, 3}, {1, 3}, {2, 2}}});
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), l0);
    mm->add_return({ret});

    migraphx::onnx_options options;
    // Parser converts the dynamic input shape to static unless there is at least one non-fixed
    // dynamic dimension. Slicing is not allowed along the non-fixed axis 1.
    options.map_dyn_input_dims["0"] = {{3, 3}, {1, 3}, {2, 2}};
    auto prog                       = migraphx::parse_onnx("slice_dyn_test.onnx", options);

    EXPECT(p == prog);
}
