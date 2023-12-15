
#include <onnx_test.hpp>

TEST_CASE(lpnormalization_default_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<std::size_t> input_lens{3, 4};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape s{input_type, input_lens};
    auto x = mm->add_parameter("x", s);

    std::ptrdiff_t axis = 0;
    auto p_val          = mm->add_instruction(migraphx::make_op("mul"), x, x);
    auto norms = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {axis}}}), p_val);
    norms      = mm->add_instruction(migraphx::make_op("sqrt"), norms);
    norms =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), norms);
    auto zero_mb =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {0.}}));
    auto one_mb =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                            mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1.}}));
    auto is_zero = mm->add_instruction(migraphx::make_op("equal"), norms, zero_mb);
    auto norms_zeros_to_one =
        mm->add_instruction(migraphx::make_op("where"), is_zero, one_mb, norms);
    mm->add_instruction(migraphx::make_op("div"), x, norms_zeros_to_one);

    auto prog = optimize_onnx("lpnormalization_default_test.onnx");
    EXPECT(p == prog);
}
