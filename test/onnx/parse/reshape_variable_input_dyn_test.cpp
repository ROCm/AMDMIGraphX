
#include <onnx_test.hpp>


TEST_CASE(reshape_variable_input_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto p0  = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}, {3, 3}}});
    auto p1    = mm->add_parameter("1", migraphx::shape{migraphx::shape::int64_type, {2}});
    auto alloc = mm->add_instruction(
        migraphx::make_op("allocate", {{"buf_type", migraphx::shape::float_type}}), p1);
    auto reshape = mm->add_instruction(migraphx::make_op("reshape"), p0, alloc);
    mm->add_return({reshape});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = parse_onnx("reshape_variable_input_dyn_test.onnx", options);
    EXPECT(p == prog);
}


