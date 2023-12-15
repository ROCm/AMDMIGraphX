
#include <onnx_test.hpp>

TEST_CASE(shrink_hard_test)
{
    migraphx::program p;
    float bias  = 0.0;
    float lambd = 1.5;
    std::vector<size_t> lens{5};
    auto* mm           = p.get_main_module();
    auto x             = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, lens});
    auto lit_bias      = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {bias}});
    auto lit_neg_lambd = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {-lambd}});
    auto lit_lambd     = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {lambd}});

    auto x_plus_bias = add_common_op(*mm, migraphx::make_op("add"), {x, lit_bias});
    auto x_min_bias  = add_common_op(*mm, migraphx::make_op("sub"), {x, lit_bias});

    auto cond1   = add_common_op(*mm, migraphx::make_op("less"), {x, lit_neg_lambd});
    auto cond2_a = add_common_op(*mm, migraphx::make_op("not"), {cond1});
    auto cond2_b = add_common_op(*mm, migraphx::make_op("greater"), {x, lit_lambd});
    auto cond2   = add_common_op(*mm, migraphx::make_op("logical_and"), {cond2_a, cond2_b});

    auto mul1 = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), cond1);
    auto mul2 = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), cond2);

    auto first  = add_common_op(*mm, migraphx::make_op("mul"), {mul1, x_plus_bias});
    auto second = add_common_op(*mm, migraphx::make_op("mul"), {mul2, x_min_bias});
    add_common_op(*mm, migraphx::make_op("add"), {first, second});
    auto prog = optimize_onnx("shrink_hard_test.onnx");
    EXPECT(p == prog);
}
