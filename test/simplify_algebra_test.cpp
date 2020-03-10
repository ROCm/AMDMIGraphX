#include <migraphx/simplify_algebra.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(simplify_add1)
{
    migraphx::program p1;
    {
        auto x    = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, x, one);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, y, two);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, x, y);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum2, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add2)
{
    migraphx::program p1;
    {
        auto x    = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, x);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, two, y);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, x, y);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum2, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add3)
{
    migraphx::program p1;
    {
        auto x    = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, x);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, one, two);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, one, sum1);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, x, sum2);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add_broadcast1)
{
    migraphx::shape inner{migraphx::shape::int32_type, {2}};
    migraphx::shape outer{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::op::broadcast b{1, {1, 2, 3, 3}};
    migraphx::program p1;
    {
        auto x    = p1.add_parameter("x", outer);
        auto y    = p1.add_parameter("y", outer);
        auto one  = p1.add_literal({inner, {1, 1}});
        auto oneb = p1.add_instruction(b, one);
        auto two  = p1.add_literal({inner, {2, 2}});
        auto twob = p1.add_instruction(b, two);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, x, oneb);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, y, twob);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum1, sum2);
        p1.add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto x     = p2.add_parameter("x", outer);
        auto y     = p2.add_parameter("y", outer);
        auto one   = p2.add_literal({inner, {1, 1}});
        auto two   = p2.add_literal({inner, {2, 2}});
        auto sum1  = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum1b = p2.add_instruction(b, sum1);
        auto sum2  = p2.add_instruction(migraphx::op::add{}, x, y);
        auto sum3  = p2.add_instruction(migraphx::op::add{}, sum2, sum1b);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add_broadcast2)
{
    migraphx::shape inner{migraphx::shape::int32_type, {2}};
    migraphx::shape outer{migraphx::shape::int32_type, {1, 2, 3, 3}};
    migraphx::op::broadcast b{1, {1, 2, 3, 3}};
    auto create_program = [&] {
        migraphx::program p;
        auto x    = p.add_parameter("x", outer);
        auto y    = p.add_parameter("y", outer);
        auto one  = p.add_literal({inner, {1, 1}});
        auto oneb = p.add_instruction(b, one);
        auto two  = p.add_literal({outer, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}});
        auto sum1 = p.add_instruction(migraphx::op::add{}, x, y);
        auto sum2 = p.add_instruction(migraphx::op::add{}, oneb, two);
        auto sum3 = p.add_instruction(migraphx::op::add{}, sum2, sum1);
        p.add_instruction(pass_op{}, sum3);
        return p;
    };
    migraphx::program p1 = create_program();
    run_pass(p1);

    migraphx::program p2 = create_program();
    EXPECT(p1 == p2);
}

// TODO: Add test case
// TEST_CASE(simplify_add4)
void simplify_add4()
{
    migraphx::program p1;
    {
        auto x    = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p1.add_literal(1);
        auto two  = p1.add_literal(2);
        auto sum1 = p1.add_instruction(migraphx::op::add{}, one, x);
        auto sum2 = p1.add_instruction(migraphx::op::add{}, sum1, y);
        auto sum3 = p1.add_instruction(migraphx::op::add{}, sum2, two);
        p1.add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto sum1 = p2.add_instruction(migraphx::op::add{}, one, two);
        auto sum2 = p2.add_instruction(migraphx::op::add{}, x, y);
        auto sum3 = p2.add_instruction(migraphx::op::add{}, sum2, sum1);
        p2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_mul_conv1)
{
    migraphx::program p;
    auto x = p.add_parameter("x", {migraphx::shape::int32_type, {1, 128, 28, 28}});
    auto w =
        p.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256, 128, 3, 3}}));
    auto conv = p.add_instruction(migraphx::op::convolution{{1, 1}, {2, 2}, {1, 1}}, x, w);
    auto a    = p.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256}}));
    auto b    = p.add_instruction(migraphx::op::broadcast{1, {1, 256, 14, 14}}, a);
    auto mul  = p.add_instruction(migraphx::op::mul{}, conv, b);
    p.add_instruction(pass_op{}, mul);
    EXPECT(conv->outputs().front()->name() == "mul");
    run_pass(p);
    auto new_conv =
        std::find_if(p.begin(), p.end(), [](auto&& ins) { return ins.name() == "convolution"; });
    EXPECT(new_conv->outputs().front()->name() != "mul");
}

TEST_CASE(simplify_mul_add)
{
    migraphx::program p1;
    {
        auto x   = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one = p1.add_literal(1);
        auto two = p1.add_literal(2);
        auto sum = p1.add_instruction(migraphx::op::add{}, one, x);
        auto mul = p1.add_instruction(migraphx::op::mul{}, sum, two);
        p1.add_instruction(pass_op{}, mul);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = p2.add_literal(1);
        auto two  = p2.add_literal(2);
        auto mul1 = p2.add_instruction(migraphx::op::mul{}, two, x);
        auto mul2 = p2.add_instruction(migraphx::op::mul{}, two, one);
        auto sum  = p2.add_instruction(migraphx::op::add{}, mul1, mul2);
        p2.add_instruction(pass_op{}, sum);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_inner_broadcast)
{
    auto b = migraphx::op::broadcast{1, {2, 1, 4, 5}};
    migraphx::program p1;
    {
        auto x   = p1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y   = p1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto xb  = p1.add_instruction(b, x);
        auto yb  = p1.add_instruction(b, y);
        auto sum = p1.add_instruction(migraphx::op::add{}, xb, yb);
        p1.add_instruction(pass_op{}, sum);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto x    = p2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = p2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto sum  = p2.add_instruction(migraphx::op::add{}, x, y);
        auto sumb = p2.add_instruction(b, sum);
        p2.add_instruction(pass_op{}, sumb);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add_conv1)
{
    migraphx::program p;
    auto x = p.add_parameter("x", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto w =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 3, 3}}));
    auto y = p.add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto v =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 3, 3}}));
    auto conv1 = p.add_instruction(migraphx::op::convolution{}, x, w);
    auto conv2 = p.add_instruction(migraphx::op::convolution{}, y, v);
    auto sum   = p.add_instruction(migraphx::op::add{}, conv1, conv2);
    p.add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    EXPECT(std::count_if(
               p.begin(), p.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 1);
}

TEST_CASE(simplify_add_conv_no_fusion_7x7_diff_strides)
{
    migraphx::program p;
    auto x = p.add_parameter("x", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto w =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 7, 7}}));
    auto y = p.add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto v =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 7, 7}}));
    auto conv1 = p.add_instruction(migraphx::op::convolution{}, x, w);
    auto conv2 = p.add_instruction(migraphx::op::convolution{{0, 0}, {3, 3}}, y, v);
    auto sum   = p.add_instruction(migraphx::op::add{}, conv1, conv2);
    p.add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    // No fusion
    EXPECT(std::count_if(
               p.begin(), p.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 2);
}

TEST_CASE(simplify_add_conv_1x1_diff_strides1)
{
    migraphx::program p;
    auto x = p.add_parameter("x", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto w =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = p.add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto v =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = p.add_instruction(migraphx::op::convolution{}, x, w);
    auto conv2 = p.add_instruction(migraphx::op::convolution{{0, 0}, {2, 2}}, y, v);
    auto sum   = p.add_instruction(migraphx::op::add{}, conv1, conv2);
    p.add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    EXPECT(std::count_if(
               p.begin(), p.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 1);
}

TEST_CASE(simplify_add_conv_1x1_diff_strides2)
{
    migraphx::program p;
    auto x = p.add_parameter("x", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto w =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = p.add_parameter("y", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto v =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = p.add_instruction(migraphx::op::convolution{{0, 0}, {2, 2}}, x, w);
    auto conv2 = p.add_instruction(migraphx::op::convolution{}, y, v);
    auto sum   = p.add_instruction(migraphx::op::add{}, conv1, conv2);
    p.add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    EXPECT(std::count_if(
               p.begin(), p.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 1);
}

TEST_CASE(simplify_add_conv_no_fusion_asymetrical_strides1)
{
    migraphx::program p;
    auto x = p.add_parameter("x", {migraphx::shape::float_type, {1, 128, 28, 14}});
    auto w =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = p.add_parameter("y", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto v =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = p.add_instruction(migraphx::op::convolution{{0, 0}, {2, 1}}, x, w);
    auto conv2 = p.add_instruction(migraphx::op::convolution{}, y, v);
    auto sum   = p.add_instruction(migraphx::op::add{}, conv1, conv2);
    p.add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    // No fusion
    EXPECT(std::count_if(
               p.begin(), p.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 2);
}

TEST_CASE(simplify_add_conv_no_fusion_asymetrical_strides2)
{
    migraphx::program p;
    auto x = p.add_parameter("x", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto w =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = p.add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 14}});
    auto v =
        p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = p.add_instruction(migraphx::op::convolution{}, x, w);
    auto conv2 = p.add_instruction(migraphx::op::convolution{{0, 0}, {2, 1}}, y, v);
    auto sum   = p.add_instruction(migraphx::op::add{}, conv1, conv2);
    p.add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    // No fusion
    EXPECT(std::count_if(
               p.begin(), p.end(), [](auto&& ins) { return ins.name() == "convolution"; }) == 2);
}

TEST_CASE(simplify_concat_add_relu)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {1}};
    migraphx::program p1;
    {
        auto x      = p1.add_parameter("x", s);
        auto y      = p1.add_parameter("y", s);
        auto one    = p1.add_literal({s, {1}});
        auto two    = p1.add_literal({s, {2}});
        auto sum1   = p1.add_instruction(migraphx::op::add{}, x, one);
        auto relu1  = p1.add_instruction(migraphx::op::relu{}, sum1);
        auto sum2   = p1.add_instruction(migraphx::op::add{}, y, two);
        auto relu2  = p1.add_instruction(migraphx::op::relu{}, sum2);
        auto concat = p1.add_instruction(migraphx::op::concat{0}, relu1, relu2);
        p1.add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto x       = p2.add_parameter("x", s);
        auto y       = p2.add_parameter("y", s);
        auto one     = p2.add_literal({s, {1}});
        auto two     = p2.add_literal({s, {2}});
        auto concat1 = p2.add_instruction(migraphx::op::concat{0}, x, y);
        auto concat2 = p2.add_instruction(migraphx::op::concat{0}, one, two);
        auto sum     = p2.add_instruction(migraphx::op::add{}, concat1, concat2);
        auto relu    = p2.add_instruction(migraphx::op::relu{}, sum);
        p2.add_instruction(pass_op{}, relu);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_concat_add_relu_broadcast_different_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {2, 1, 4, 5}};
    migraphx::program p1;
    {
        auto b      = migraphx::op::broadcast{1, {2, 1, 4, 5}};
        auto x      = p1.add_parameter("x", s);
        auto y      = p1.add_parameter("y", s);
        auto one    = p1.add_literal(1);
        auto oneb   = p1.add_instruction(b, one);
        auto two    = p1.add_literal(2);
        auto twob   = p1.add_instruction(b, two);
        auto sum1   = p1.add_instruction(migraphx::op::add{}, x, oneb);
        auto relu1  = p1.add_instruction(migraphx::op::relu{}, sum1);
        auto sum2   = p1.add_instruction(migraphx::op::add{}, y, twob);
        auto relu2  = p1.add_instruction(migraphx::op::relu{}, sum2);
        auto concat = p1.add_instruction(migraphx::op::concat{1}, relu1, relu2);
        p1.add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto b        = migraphx::op::broadcast{1, {2, 2, 4, 5}};
        auto x        = p2.add_parameter("x", s);
        auto y        = p2.add_parameter("y", s);
        auto one      = p2.add_literal(1);
        auto two      = p2.add_literal(2);
        auto concat1  = p2.add_instruction(migraphx::op::concat{1}, x, y);
        auto concat2  = p2.add_instruction(migraphx::op::concat{0}, one, two);
        auto concat2b = p2.add_instruction(b, concat2);
        auto sum      = p2.add_instruction(migraphx::op::add{}, concat1, concat2b);
        auto relu     = p2.add_instruction(migraphx::op::relu{}, sum);
        p2.add_instruction(pass_op{}, relu);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_concat_add_relu_broadcast_same_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {2, 1, 4, 5}};
    migraphx::program p1;
    {
        auto b      = migraphx::op::broadcast{1, {2, 1, 4, 5}};
        auto x      = p1.add_parameter("x", s);
        auto y      = p1.add_parameter("y", s);
        auto one    = p1.add_literal(1);
        auto oneb   = p1.add_instruction(b, one);
        auto two    = p1.add_literal(2);
        auto twob   = p1.add_instruction(b, two);
        auto sum1   = p1.add_instruction(migraphx::op::add{}, x, oneb);
        auto relu1  = p1.add_instruction(migraphx::op::relu{}, sum1);
        auto sum2   = p1.add_instruction(migraphx::op::add{}, y, twob);
        auto relu2  = p1.add_instruction(migraphx::op::relu{}, sum2);
        auto concat = p1.add_instruction(migraphx::op::concat{0}, relu1, relu2);
        p1.add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto b       = migraphx::op::broadcast{1, {2, 1, 4, 5}};
        auto x       = p2.add_parameter("x", s);
        auto y       = p2.add_parameter("y", s);
        auto one     = p2.add_literal(1);
        auto oneb    = p2.add_instruction(b, one);
        auto two     = p2.add_literal(2);
        auto twob    = p2.add_instruction(b, two);
        auto concat1 = p2.add_instruction(migraphx::op::concat{0}, x, y);
        auto concat2 = p2.add_instruction(migraphx::op::concat{0}, oneb, twob);
        auto sum     = p2.add_instruction(migraphx::op::add{}, concat1, concat2);
        auto relu    = p2.add_instruction(migraphx::op::relu{}, sum);
        p2.add_instruction(pass_op{}, relu);
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
