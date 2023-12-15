
#include <onnx_test.hpp>


TEST_CASE(binary_dyn_brcst_prelu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}, {5, 5}}});
    auto l1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 5}});

    auto ret = add_common_op(*mm, migraphx::make_op("prelu"), {l0, l1});
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog = migraphx::parse_onnx("binary_dyn_brcst_prelu_test.onnx", options);

    EXPECT(p == prog);
}


