
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>


TEST_CASE(globallppool_dyn_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {16, 32}, {16, 32}}});
    auto ret = mm->add_instruction(migraphx::make_op("pooling",
                                                     {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                      {"dyn_global", true},
                                                      {"padding", {0, 0, 0, 0}},
                                                      {"lengths", {}}}),
                                   input);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {16, 32};
    auto prog                     = migraphx::parse_onnx("globallppool_dyn_test.onnx", options);

    EXPECT(p == prog);
}


