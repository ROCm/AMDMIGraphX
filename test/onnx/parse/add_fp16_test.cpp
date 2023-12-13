
#include <onnx_test.hpp>

TEST_CASE(add_fp16_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {1.5}});
    auto l1 =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {2.5}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    auto prog = optimize_onnx("add_fp16_test.onnx");

    EXPECT(p == prog);
}
