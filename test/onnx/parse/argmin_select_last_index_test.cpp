
#include <onnx_test.hpp>


TEST_CASE(argmin_select_last_index_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto ins = mm->add_instruction(
        migraphx::make_op("argmin", {{"axis", 3}, {"select_last_index", true}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), ins);
    auto prog = optimize_onnx("argmin_select_last_index_test.onnx");

    EXPECT(p == prog);
}


