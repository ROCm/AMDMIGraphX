
#include <onnx_test.hpp>

TEST_CASE(conv_transpose_dyn_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_parameter("x", {migraphx::shape::float_type, {{1, 4}, {1, 1}, {3, 3}, {3, 3}}});
    auto l1  = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto ret = mm->add_instruction(migraphx::make_op("convolution_backwards"), l0, l1);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = parse_onnx("conv_transpose_dyn_batch_test.onnx", options);
    EXPECT(p == prog);
}
