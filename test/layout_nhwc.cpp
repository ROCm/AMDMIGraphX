#include <migraphx/layout_nhwc.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::layout_nhwc{}, migraphx::dead_code_elimination{}});
}

migraphx::operation layout(std::vector<int64_t> permutation = {0, 1, 2, 3})
{
    return migraphx::make_op("layout", {{"permutation", permutation}});
}

migraphx::instruction_ref add_layout_nhwc(migraphx::module& m, migraphx::instruction_ref ins)
{
    return m.add_instruction(layout({0, 2, 3, 1}), ins);
}

TEST_CASE(conv_relu)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 3, 3}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        m1.add_instruction(migraphx::make_op("relu"), conv);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w    = add_layout_nhwc(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::float_type, {16, 8, 3, 3}})));
        auto conv = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), conv);
        m2.add_instruction(layout(), relu);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(conv_add)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 3, 3}}));
        auto y    = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {16}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        m1.add_instruction(migraphx::make_op("add"), conv, b);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w    = add_layout_nhwc(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::float_type, {16, 8, 3, 3}})));
        auto y    = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {16}}));
        auto conv = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        auto add = m2.add_instruction(migraphx::make_op("add"), conv, b);
        m2.add_instruction(layout(), add);
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
