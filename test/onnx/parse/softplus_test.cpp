
#include <onnx_test.hpp>

TEST_CASE(softplus_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<std::size_t> input_lens{5};
    auto input_type = migraphx::shape::float_type;

    auto x = mm->add_parameter("x", migraphx::shape{input_type, input_lens});
    auto mb_ones =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));
    auto exp = mm->add_instruction(migraphx::make_op("exp"), x);
    auto add = mm->add_instruction(migraphx::make_op("add"), exp, mb_ones);
    mm->add_instruction(migraphx::make_op("log"), add);

    auto prog = optimize_onnx("softplus_test.onnx");
    EXPECT(p == prog);
}
