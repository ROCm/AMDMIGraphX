
#include <onnx_test.hpp>

TEST_CASE(gather_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}, {5, 5}, {6, 6}}});
    auto l1 = mm->add_parameter(
        "indices", migraphx::shape{migraphx::shape::int32_type, {{1, 4}, {3, 3}, {4, 4}, {5, 5}}});
    auto cont_l0 = mm->add_instruction(migraphx::make_op("contiguous"), l0);
    auto cont_l1 = mm->add_instruction(migraphx::make_op("contiguous"), l1);

    int axis       = 1;
    auto gather_op = migraphx::make_op("gather", {{"axis", axis}});
    auto ret       = mm->add_instruction(gather_op, cont_l0, cont_l1);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = parse_onnx("gather_dyn_test.onnx", options);

    EXPECT(p == prog);
}
