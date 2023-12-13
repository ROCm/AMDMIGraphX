
#include <onnx_test.hpp>


TEST_CASE(argmax_select_last_index_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto ins = mm->add_instruction(
        migraphx::make_op("argmax", {{"axis", 2}, {"select_last_index", true}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), ins);
    auto prog = optimize_onnx("argmax_select_last_index_test.onnx");

    EXPECT(p == prog);
}


