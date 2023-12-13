
#include <onnx_test.hpp>
#include <migraphx/op/lrn.hpp>


TEST_CASE(lrn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 28, 24, 24}});
    migraphx::op::lrn op;
    op.size  = 5;
    op.alpha = 0.0001;
    op.beta  = 0.75;
    op.bias  = 1.0;
    mm->add_instruction(op, l0);
    auto prog = optimize_onnx("lrn_test.onnx");

    EXPECT(p == prog);
}


