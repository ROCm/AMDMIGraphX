
#include <onnx_test.hpp>

TEST_CASE(flatten_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}, {5, 5}}});
    auto c0  = mm->add_instruction(migraphx::make_op("contiguous"), l0);
    auto ret = mm->add_instruction(migraphx::make_op("flatten", {{"axis", 2}}), c0);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = parse_onnx("flatten_dyn_test.onnx", options);
    EXPECT(p == prog);
}
