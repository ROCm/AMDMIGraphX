
#include <onnx_test.hpp>


TEST_CASE(argmax_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}, {5, 5}, {6, 6}}});
    auto ins = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), l0);
    auto ret = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), ins);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = parse_onnx("argmax_dyn_test.onnx", options);

    EXPECT(p == prog);
}


