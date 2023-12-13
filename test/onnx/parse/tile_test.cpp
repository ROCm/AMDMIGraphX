
#include <onnx_test.hpp>

TEST_CASE(tile_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, {1, 2}});
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), input, input);

    auto prog = optimize_onnx("tile_test.onnx");

    EXPECT(p == prog);
}
