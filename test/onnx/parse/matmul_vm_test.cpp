
#include <onnx_test.hpp>
#include <migraphx/apply_alpha_beta.hpp>


TEST_CASE(matmul_vm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7, 8}});
    auto sl0 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l0);
    auto res = migraphx::add_apply_alpha_beta(*mm, {sl0, l1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), res);

    auto prog = optimize_onnx("matmul_vm_test.onnx");

    EXPECT(p == prog);
}


