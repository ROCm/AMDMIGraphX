
#include <onnx_test.hpp>
#include <migraphx/apply_alpha_beta.hpp>


TEST_CASE(matmul_bmbm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1 = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {5, 2, 1, 7, 8}});
    auto bl0 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {5, 2, 3, 6, 7}}}), l0);
    auto bl1 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {5, 2, 3, 7, 8}}}), l1);
    migraphx::add_apply_alpha_beta(*mm, {bl0, bl1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    auto prog = optimize_onnx("matmul_bmbm_test.onnx");

    EXPECT(p == prog);
}


