
#include <onnx_test.hpp>

TEST_CASE(concat_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 4, 3}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7, 4, 3}});
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), l0, l1);
    auto prog = optimize_onnx("concat_test.onnx");

    EXPECT(p == prog);
}
