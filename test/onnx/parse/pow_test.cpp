
#include <onnx_test.hpp>

TEST_CASE(pow_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    mm->add_instruction(migraphx::make_op("pow"), l0, l1);

    auto prog = optimize_onnx("pow_test.onnx");

    EXPECT(p == prog);
}
