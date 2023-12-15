
#include <onnx_test.hpp>


TEST_CASE(tile_test_3x2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, {3, 2}});
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    auto l0    = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), input, input);
    auto l1    = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), l0, input);
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l1, l1);

    auto prog = optimize_onnx("tile_test_3x2.onnx");

    EXPECT(p == prog);
}


