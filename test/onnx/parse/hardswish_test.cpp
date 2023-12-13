
#include <onnx_test.hpp>


TEST_CASE(hardswish_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<std::size_t> input_lens{2, 5};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape s{input_type, input_lens};
    auto x = mm->add_parameter("x", s);

    float alpha = 1.0 / 6.0;
    float beta  = 0.5;

    auto mb_alpha = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
        mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {alpha}}));
    auto mb_beta = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
        mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {beta}}));
    auto mb_zero =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}));
    auto mb_one =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));

    auto mul         = mm->add_instruction(migraphx::make_op("mul"), mb_alpha, x);
    auto add         = mm->add_instruction(migraphx::make_op("add"), mb_beta, mul);
    auto hardsigmoid = mm->add_instruction(migraphx::make_op("clip"), add, mb_zero, mb_one);
    mm->add_instruction(migraphx::make_op("mul"), x, hardsigmoid);

    auto prog = optimize_onnx("hardswish_test.onnx");

    EXPECT(p == prog);
}


