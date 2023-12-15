
#include <onnx_test.hpp>
#include <migraphx/apply_alpha_beta.hpp>

TEST_CASE(matmul_dyn_mv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {{4, 8, {6}}, {7, 7}}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto res = migraphx::add_apply_alpha_beta(*mm, {l0, sl1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    auto ret = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), res);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["1"] = {{4, 8, {6}}, {7, 7}};
    auto prog                       = parse_onnx("matmul_dyn_mv_test.onnx", options);

    EXPECT(p == prog);
}
