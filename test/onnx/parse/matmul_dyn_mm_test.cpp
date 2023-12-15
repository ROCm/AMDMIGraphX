
#include <onnx_test.hpp>
#include <migraphx/apply_alpha_beta.hpp>

TEST_CASE(matmul_dyn_mm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {{4, 8, {6}}, {7, 7}}});
    auto l1 =
        mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {{7, 7}, {1, 5, {3}}}});
    auto ret = migraphx::add_apply_alpha_beta(*mm, {l0, l1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["1"] = {{4, 8, {6}}, {7, 7}};
    options.map_dyn_input_dims["2"] = {{7, 7}, {1, 5, {3}}};
    auto prog                       = parse_onnx("matmul_dyn_mm_test.onnx", options);

    EXPECT(p == prog);
}
