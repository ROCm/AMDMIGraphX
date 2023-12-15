
#include <onnx_test.hpp>
#include <migraphx/apply_alpha_beta.hpp>


TEST_CASE(matmul_dyn_vv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{5, 8, {7}};
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {dd}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {dd}});
    auto sl0 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l0);
    auto sl1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto res =
        migraphx::add_apply_alpha_beta(*mm, {sl0, sl1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    auto sr0 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), res);
    auto ret = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sr0);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = dd;
    auto prog                     = parse_onnx("matmul_dyn_vv_test.onnx", options);

    EXPECT(p == prog);
}


