
#include <onnx_test.hpp>


TEST_CASE(conv_dynamic_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_parameter("0", {migraphx::shape::float_type, {{1, 6}, {3, 3}, {5, 5}, {5, 5}}});
    auto l1 = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 3, 3}});
    auto c0 = mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
        l0,
        l1);
    mm->add_return({c0});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 6};

    auto prog = migraphx::parse_onnx("conv_dynamic_batch_test.onnx", options);
    EXPECT(p == prog);
}


