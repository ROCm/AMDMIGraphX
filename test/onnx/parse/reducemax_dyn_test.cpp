
#include <onnx_test.hpp>


TEST_CASE(reducemax_dyn_test)
{
    // input shape with 4 dynamic dimensions
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {{3, 5}, {4, 4}, {5, 5}, {6, 6}}});
    auto r0 = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), l0);
    auto r1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), r0);
    mm->add_return({r1});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = {{3, 5}, {4, 4}, {5, 5}, {6, 6}};
    auto prog                       = migraphx::parse_onnx("reducemax_dyn_test.onnx", options);

    EXPECT(p == prog);
}


