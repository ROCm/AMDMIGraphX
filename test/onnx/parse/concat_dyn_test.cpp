
#include <onnx_test.hpp>


TEST_CASE(concat_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {1, 4}, {3, 3}}});
    auto l1 = mm->add_parameter(
        "1", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {1, 4}, {3, 3}}});
    auto ret = mm->add_instruction(migraphx::make_op("concat"), l0, l1);

    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = parse_onnx("concat_dyn_test.onnx", options);

    EXPECT(p == prog);
}


