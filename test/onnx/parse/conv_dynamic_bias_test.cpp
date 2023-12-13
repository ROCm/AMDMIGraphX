
#include <onnx_test.hpp>

TEST_CASE(conv_dynamic_bias_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x0 =
        mm->add_parameter("0", {migraphx::shape::float_type, {{1, 6}, {3, 3}, {32, 32}, {32, 32}}});
    auto x1 = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto x2 = mm->add_parameter("2", {migraphx::shape::float_type, {1}});
    auto x3 = mm->add_instruction(migraphx::make_op("convolution"), x0, x1);
    auto x4 = mm->add_instruction(migraphx::make_op("broadcast", {{"axis", 1}}), x2, x3);
    auto x5 = mm->add_instruction(migraphx::make_op("add"), x3, x4);
    mm->add_return({x5});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 6};
    auto prog                     = migraphx::parse_onnx("conv_dynamic_bias_test.onnx", options);
    EXPECT(p == prog);
}
