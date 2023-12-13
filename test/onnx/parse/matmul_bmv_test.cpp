
#include <onnx_test.hpp>
#include <migraphx/apply_alpha_beta.hpp>


TEST_CASE(matmul_bmv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto sl1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto bsl1 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 7, 1}}}), sl1);
    auto res =
        migraphx::add_apply_alpha_beta(*mm, {l0, bsl1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), res);

    auto prog = optimize_onnx("matmul_bmv_test.onnx");

    EXPECT(p == prog);
}


