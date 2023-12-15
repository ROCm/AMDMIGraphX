
#include <onnx_test.hpp>


TEST_CASE(hardsigmoid_double_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<std::size_t> input_lens{1, 3, 4, 5};
    auto input_type = migraphx::shape::double_type;
    migraphx::shape s{input_type, input_lens};
    auto x = mm->add_parameter("x", s);

    float alpha = 0.3;
    float beta  = 0.7;

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

    auto mul = mm->add_instruction(migraphx::make_op("mul"), mb_alpha, x);
    auto add = mm->add_instruction(migraphx::make_op("add"), mb_beta, mul);
    mm->add_instruction(migraphx::make_op("clip"), add, mb_zero, mb_one);

    auto prog = optimize_onnx("hardsigmoid_double_test.onnx");
    EXPECT(p == prog);
}


