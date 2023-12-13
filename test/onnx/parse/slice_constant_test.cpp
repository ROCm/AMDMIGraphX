
#include <onnx_test.hpp>


TEST_CASE(slice_constant_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::float_type, {3, 2}}, {0, 1, 2, 3, 4, 5}});
    mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {1, 0}}, {"ends", {2, 2}}}), l0);
    auto prog = optimize_onnx("slice_constant_test.onnx");

    EXPECT(p == prog);
}


