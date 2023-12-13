
#include <onnx_test.hpp>

TEST_CASE(flatten_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    mm->add_instruction(migraphx::make_op("flatten", {{"axis", 2}}), l0);
    mm->add_instruction(migraphx::make_op("flatten", {{"axis", 1}}), l0);
    auto prog = optimize_onnx("flatten_test.onnx");

    EXPECT(p == prog);
}
