
#include <onnx_test.hpp>


TEST_CASE(gemm_dyn_bias_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x0 =
        mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {{8, 8}, {1, 10}}});
    auto x1   = mm->add_parameter("B", migraphx::shape{migraphx::shape::float_type, {8, 7}});
    auto x2   = mm->add_parameter("C", migraphx::shape{migraphx::shape::float_type, {1, 7}});
    auto x0_t = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x0);
    auto dot  = mm->add_instruction(migraphx::make_op("dot"), x0_t, x1);
    auto x2_b = mm->add_instruction(migraphx::make_op("multibroadcast"), x2, dot);
    auto ret  = mm->add_instruction(migraphx::make_op("add"), dot, x2_b);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 10};
    auto prog                     = parse_onnx("gemm_dyn_bias_test.onnx", options);
    EXPECT(p == prog);
}


