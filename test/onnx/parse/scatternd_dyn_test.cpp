
#include <onnx_test.hpp>


TEST_CASE(scatternd_dyn_test)
{
    // dynamic input.
    migraphx::program p;
    auto* mm = p.get_main_module();
    // parameters with dynamic dimensions
    auto l0 = mm->add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {{1, 3, {2}}, {2, 2}, {2, 2}}});
    auto l1 = mm->add_parameter(
        "indices", migraphx::shape{migraphx::shape::int64_type, {{2, 1, {2}}, {1, 1}, {2, 2}}});
    auto l2 = mm->add_parameter(
        "updates", migraphx::shape{migraphx::shape::float_type, {{2, 1, {2}}, {1, 1}, {2, 2}}});
    auto r = mm->add_instruction(migraphx::make_op("scatternd_none"), l0, l1, l2);
    mm->add_return({r});
    migraphx::onnx_options options;
    options.map_dyn_input_dims["data"]    = {{1, 3, {2}}, {2, 2}, {2, 2}};
    options.map_dyn_input_dims["indices"] = {{2, 1, {2}}, {1, 1}, {2, 2}};
    options.map_dyn_input_dims["updates"] = {{2, 1, {2}}, {1, 1}, {2, 2}};
    auto prog = migraphx::parse_onnx("scatternd_dyn_test.onnx", options);

    EXPECT(p == prog);
}


