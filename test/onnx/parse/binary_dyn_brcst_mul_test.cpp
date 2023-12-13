
#include <onnx_test.hpp>

TEST_CASE(binary_dyn_brcst_mul_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}, {5, 5}}});
    auto l1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 1}});

    auto bl1 = mm->add_instruction(
        migraphx::make_op("multibroadcast",
                          {{"out_dyn_dims", to_value(l0->get_shape().dyn_dims())}}),
        l1,
        l0);
    auto ret = mm->add_instruction(migraphx::make_op("mul"), l0, bl1);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = migraphx::parse_onnx("binary_dyn_brcst_mul_test.onnx", options);

    EXPECT(p == prog);
}
