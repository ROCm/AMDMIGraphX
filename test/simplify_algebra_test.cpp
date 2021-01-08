#include <migraphx/simplify_algebra.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(*p.get_main_module(),
                         {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(simplify_add1)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = mm1->add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = mm1->add_literal(1);
        auto two  = mm1->add_literal(2);
        auto sum1 = mm1->add_instruction(migraphx::make_op("add"), x, one);
        auto sum2 = mm1->add_instruction(migraphx::make_op("add"), y, two);
        auto sum3 = mm1->add_instruction(migraphx::make_op("add"), sum1, sum2);
        mm1->add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = mm2->add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = mm2->add_literal(1);
        auto two  = mm2->add_literal(2);
        auto sum1 = mm2->add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = mm2->add_instruction(migraphx::make_op("add"), x, y);
        auto sum3 = mm2->add_instruction(migraphx::make_op("add"), sum2, sum1);
        mm2->add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add2)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = mm1->add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = mm1->add_literal(1);
        auto two  = mm1->add_literal(2);
        auto sum1 = mm1->add_instruction(migraphx::make_op("add"), one, x);
        auto sum2 = mm1->add_instruction(migraphx::make_op("add"), two, y);
        auto sum3 = mm1->add_instruction(migraphx::make_op("add"), sum1, sum2);
        mm1->add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = mm2->add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = mm2->add_literal(1);
        auto two  = mm2->add_literal(2);
        auto sum1 = mm2->add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = mm2->add_instruction(migraphx::make_op("add"), x, y);
        auto sum3 = mm2->add_instruction(migraphx::make_op("add"), sum2, sum1);
        mm2->add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add3)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = mm1->add_literal(1);
        auto two  = mm1->add_literal(2);
        auto sum1 = mm1->add_instruction(migraphx::make_op("add"), one, x);
        auto sum2 = mm1->add_instruction(migraphx::make_op("add"), one, two);
        auto sum3 = mm1->add_instruction(migraphx::make_op("add"), sum1, sum2);
        mm1->add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = mm2->add_literal(1);
        auto two  = mm2->add_literal(2);
        auto sum1 = mm2->add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = mm2->add_instruction(migraphx::make_op("add"), one, sum1);
        auto sum3 = mm2->add_instruction(migraphx::make_op("add"), x, sum2);
        mm2->add_instruction(pass_op{}, sum3);
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
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", outer);
        auto y    = mm1->add_parameter("y", outer);
        auto one  = mm1->add_literal({inner, {1, 1}});
        auto oneb = mm1->add_instruction(b, one);
        auto two  = mm1->add_literal({inner, {2, 2}});
        auto twob = mm1->add_instruction(b, two);
        auto sum1 = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto sum2 = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto sum3 = mm1->add_instruction(migraphx::make_op("add"), sum1, sum2);
        mm1->add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2  = p2.get_main_module();
        auto x     = mm2->add_parameter("x", outer);
        auto y     = mm2->add_parameter("y", outer);
        auto one   = mm2->add_literal({inner, {1, 1}});
        auto two   = mm2->add_literal({inner, {2, 2}});
        auto sum1  = mm2->add_instruction(migraphx::make_op("add"), one, two);
        auto sum1b = mm2->add_instruction(b, sum1);
        auto sum2  = mm2->add_instruction(migraphx::make_op("add"), x, y);
        auto sum3  = mm2->add_instruction(migraphx::make_op("add"), sum2, sum1b);
        mm2->add_instruction(pass_op{}, sum3);
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
        auto* mm  = p.get_main_module();
        auto x    = mm->add_parameter("x", outer);
        auto y    = mm->add_parameter("y", outer);
        auto one  = mm->add_literal({inner, {1, 1}});
        auto oneb = mm->add_instruction(b, one);
        auto two = mm->add_literal({outer, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}});
        auto sum1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto sum2 = mm->add_instruction(migraphx::make_op("add"), oneb, two);
        auto sum3 = mm->add_instruction(migraphx::make_op("add"), sum2, sum1);
        mm->add_instruction(pass_op{}, sum3);
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
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = mm1->add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = mm1->add_literal(1);
        auto two  = mm1->add_literal(2);
        auto sum1 = mm1->add_instruction(migraphx::make_op("add"), one, x);
        auto sum2 = mm1->add_instruction(migraphx::make_op("add"), sum1, y);
        auto sum3 = mm1->add_instruction(migraphx::make_op("add"), sum2, two);
        mm1->add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = mm2->add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = mm2->add_literal(1);
        auto two  = mm2->add_literal(2);
        auto sum1 = mm2->add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = mm2->add_instruction(migraphx::make_op("add"), x, y);
        auto sum3 = mm2->add_instruction(migraphx::make_op("add"), sum2, sum1);
        mm2->add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_mul_conv1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::int32_type, {1, 128, 28, 28}});
    auto w   = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::int32_type, {256, 128, 3, 3}}));
    auto conv = mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        x,
        w);
    auto a = mm->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {256}}));
    auto b = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 256, 14, 14}}}), a);
    auto mul = mm->add_instruction(migraphx::make_op("mul"), conv, b);
    mm->add_instruction(pass_op{}, mul);
    EXPECT(conv->outputs().front()->name() == "mul");
    run_pass(p);
    auto new_conv = std::find_if(
        mm->begin(), mm->end(), [](auto&& ins) { return ins.name() == "convolution"; });
    EXPECT(new_conv->outputs().front()->name() != "mul");
}

TEST_CASE(simplify_mul_slice_conv1)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w    = mm1->add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto conv   = mm1->add_instruction(migraphx::make_op("convolution"), x, w);
        auto slice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto a = mm1->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}));
        auto b = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 384, 17, 17}}}), a);
        auto mul    = mm1->add_instruction(migraphx::make_op("mul"), slice1, b);
        auto slice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {384}}, {"ends", {768}}}), conv);
        auto add = mm1->add_instruction(migraphx::make_op("add"), mul, slice2);
        mm1->add_instruction(pass_op{}, add);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w    = mm2->add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto wslice1 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {384}}}), w);
        auto a = mm2->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}));
        auto b = mm2->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"dims", {384, 1024, 1, 1}}}), a);
        auto mul     = mm2->add_instruction(migraphx::make_op("mul"), b, wslice1);
        auto wslice2 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {384}}, {"ends", {768}}}), w);
        auto concat =
            mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), mul, wslice2);
        auto conv   = mm2->add_instruction(migraphx::make_op("convolution"), x, concat);
        auto slice1 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto slice2 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {384}}, {"ends", {768}}}), conv);
        auto add = mm2->add_instruction(migraphx::make_op("add"), slice1, slice2);
        mm2->add_instruction(pass_op{}, add);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_mul_slice_conv_overlapping_slice)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w    = mm1->add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto conv   = mm1->add_instruction(migraphx::make_op("convolution"), x, w);
        auto slice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto a = mm1->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}));
        auto b = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 384, 17, 17}}}), a);
        auto mul    = mm1->add_instruction(migraphx::make_op("mul"), slice1, b);
        auto slice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {383}}, {"ends", {767}}}), conv);
        auto add = mm1->add_instruction(migraphx::make_op("add"), mul, slice2);
        mm1->add_instruction(pass_op{}, add);
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_mul_slice_conv_not_all_slice)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w    = mm1->add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto conv   = mm1->add_instruction(migraphx::make_op("convolution"), x, w);
        auto slice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto a = mm1->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}));
        auto b = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 384, 17, 17}}}), a);
        auto mul = mm1->add_instruction(migraphx::make_op("mul"), slice1, b);
        auto c   = mm1->add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {1, 768, 17, 17}}));
        auto add    = mm1->add_instruction(migraphx::make_op("add"), conv, c);
        auto concat = mm1->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), mul, add);
        mm1->add_instruction(pass_op{}, concat);
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_mul_add)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = mm1->add_literal(1);
        auto two  = mm1->add_literal(2);
        auto sum  = mm1->add_instruction(migraphx::make_op("add"), one, x);
        auto mul  = mm1->add_instruction(migraphx::make_op("mul"), sum, two);
        mm1->add_instruction(pass_op{}, mul);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = mm2->add_literal(1);
        auto two  = mm2->add_literal(2);
        auto mul1 = mm2->add_instruction(migraphx::make_op("mul"), two, x);
        auto mul2 = mm2->add_instruction(migraphx::make_op("mul"), two, one);
        auto sum  = mm2->add_instruction(migraphx::make_op("add"), mul1, mul2);
        mm2->add_instruction(pass_op{}, sum);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_inner_broadcast)
{
    auto b = migraphx::op::broadcast{1, {2, 1, 4, 5}};
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = mm1->add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto xb   = mm1->add_instruction(b, x);
        auto yb   = mm1->add_instruction(b, y);
        auto sum  = mm1->add_instruction(migraphx::make_op("add"), xb, yb);
        mm1->add_instruction(pass_op{}, sum);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = mm2->add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto sum  = mm2->add_instruction(migraphx::make_op("add"), x, y);
        auto sumb = mm2->add_instruction(b, sum);
        mm2->add_instruction(pass_op{}, sumb);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_add_conv1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto w   = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 3, 3}}));
    auto y = mm->add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto v = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 3, 3}}));
    auto conv1 = mm->add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = mm->add_instruction(migraphx::make_op("convolution"), y, v);
    auto sum   = mm->add_instruction(migraphx::make_op("add"), conv1, conv2);
    mm->add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    EXPECT(std::count_if(mm->begin(), mm->end(), [](auto&& ins) {
               return ins.name() == "convolution";
           }) == 1);
}

TEST_CASE(simplify_add_conv_no_fusion_7x7_diff_strides)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto w   = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 7, 7}}));
    auto y = mm->add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto v = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 7, 7}}));
    auto conv1 = mm->add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {3, 3}}}), y, v);
    auto sum = mm->add_instruction(migraphx::make_op("add"), conv1, conv2);
    mm->add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    // No fusion
    EXPECT(std::count_if(mm->begin(), mm->end(), [](auto&& ins) {
               return ins.name() == "convolution";
           }) == 2);
}

TEST_CASE(simplify_add_conv_1x1_diff_strides1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto w   = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = mm->add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto v = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = mm->add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 2}}}), y, v);
    auto sum = mm->add_instruction(migraphx::make_op("add"), conv1, conv2);
    mm->add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    EXPECT(std::count_if(mm->begin(), mm->end(), [](auto&& ins) {
               return ins.name() == "convolution";
           }) == 1);
}

TEST_CASE(simplify_add_conv_1x1_diff_strides2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 128, 28, 28}});
    auto w   = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = mm->add_parameter("y", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto v = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 2}}}), x, w);
    auto conv2 = mm->add_instruction(migraphx::make_op("convolution"), y, v);
    auto sum   = mm->add_instruction(migraphx::make_op("add"), conv1, conv2);
    mm->add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    EXPECT(std::count_if(mm->begin(), mm->end(), [](auto&& ins) {
               return ins.name() == "convolution";
           }) == 1);
}

TEST_CASE(simplify_add_conv_1x1_diff_strides_odd)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 54, 83, 83}});
    auto w =
        mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {54, 54, 1, 1}}));
    auto y = mm->add_parameter("y", {migraphx::shape::float_type, {1, 54, 165, 165}});
    auto v =
        mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {54, 54, 1, 1}}));
    auto conv1 = mm->add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 2}}}), y, v);
    auto sum = mm->add_instruction(migraphx::make_op("add"), conv1, conv2);
    mm->add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    EXPECT(std::count_if(mm->begin(), mm->end(), [](auto&& ins) {
               return ins.name() == "convolution";
           }) == 1);
}

TEST_CASE(simplify_add_conv_no_fusion_asymetrical_strides1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 128, 28, 14}});
    auto w   = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = mm->add_parameter("y", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto v = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 1}}}), x, w);
    auto conv2 = mm->add_instruction(migraphx::make_op("convolution"), y, v);
    auto sum   = mm->add_instruction(migraphx::make_op("add"), conv1, conv2);
    mm->add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    // No fusion
    EXPECT(std::count_if(mm->begin(), mm->end(), [](auto&& ins) {
               return ins.name() == "convolution";
           }) == 2);
}

TEST_CASE(simplify_add_conv_no_fusion_asymetrical_strides2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 128, 14, 14}});
    auto w   = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto y = mm->add_parameter("y", {migraphx::shape::float_type, {1, 128, 28, 14}});
    auto v = mm->add_literal(
        migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
    auto conv1 = mm->add_instruction(migraphx::make_op("convolution"), x, w);
    auto conv2 = mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {2, 1}}}), y, v);
    auto sum = mm->add_instruction(migraphx::make_op("add"), conv1, conv2);
    mm->add_instruction(pass_op{}, sum);
    auto s = p.get_output_shapes().back();
    run_pass(p);
    EXPECT(s == p.get_output_shapes().back());
    // No fusion
    EXPECT(std::count_if(mm->begin(), mm->end(), [](auto&& ins) {
               return ins.name() == "convolution";
           }) == 2);
}

TEST_CASE(simplify_concat_add_relu)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {1}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto x     = mm1->add_parameter("x", s);
        auto y     = mm1->add_parameter("y", s);
        auto one   = mm1->add_literal({s, {1}});
        auto two   = mm1->add_literal({s, {2}});
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, one);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, two);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto concat =
            mm1->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), relu1, relu2);
        mm1->add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto x       = mm2->add_parameter("x", s);
        auto y       = mm2->add_parameter("y", s);
        auto one     = mm2->add_literal({s, {1}});
        auto two     = mm2->add_literal({s, {2}});
        auto concat1 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto concat2 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto sum     = mm2->add_instruction(migraphx::make_op("add"), concat1, concat2);
        auto relu    = mm2->add_instruction(migraphx::make_op("relu"), sum);
        mm2->add_instruction(pass_op{}, relu);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_concat_add_relu_partial)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {1}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto x     = mm1->add_parameter("x", s);
        auto y     = mm1->add_parameter("y", s);
        auto one   = mm1->add_literal({s, {1}});
        auto two   = mm1->add_literal({s, {2}});
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, one);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, two);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto sum3  = mm1->add_instruction(migraphx::make_op("add"), x, y);
        auto concat =
            mm1->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), sum3, relu1, relu2);
        mm1->add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto x       = mm2->add_parameter("x", s);
        auto y       = mm2->add_parameter("y", s);
        auto one     = mm2->add_literal({s, {1}});
        auto two     = mm2->add_literal({s, {2}});
        auto concat1 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto concat2 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto sum1    = mm2->add_instruction(migraphx::make_op("add"), concat1, concat2);
        auto relu    = mm2->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2    = mm2->add_instruction(migraphx::make_op("add"), x, y);
        auto concat  = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), sum2, relu);
        mm2->add_instruction(pass_op{}, concat);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_concat_add_relu_partial_broadcast)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {2, 1, 4, 5}};
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto b    = migraphx::op::broadcast{1, {2, 1, 4, 5}};
        auto x    = mm1->add_parameter("x", s);
        auto y    = mm1->add_parameter("y", s);
        auto one  = mm1->add_literal(1);
        auto oneb = mm1->add_instruction(b, one);
        auto two  = mm1->add_literal(2);
        auto twob = mm1->add_instruction(b, two);
        auto sum  = mm1->add_instruction(migraphx::make_op("add"), x, y);
        auto concat =
            mm1->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), sum, oneb, twob);
        mm1->add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto b       = migraphx::op::broadcast{1, {2, 2, 4, 5}};
        auto x       = mm2->add_parameter("x", s);
        auto y       = mm2->add_parameter("y", s);
        auto one     = mm2->add_literal(1);
        auto two     = mm2->add_literal(2);
        auto concat1 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = mm2->add_instruction(b, concat1);
        auto sum     = mm2->add_instruction(migraphx::make_op("add"), x, y);
        auto concat2 =
            mm2->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), sum, concatb);
        mm2->add_instruction(pass_op{}, concat2);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_concat_add_relu_broadcast_different_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {2, 1, 4, 5}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {2, 1, 4, 5}};
        auto x     = mm1->add_parameter("x", s);
        auto y     = mm1->add_parameter("y", s);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto concat =
            mm1->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), relu1, relu2);
        mm1->add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2     = p2.get_main_module();
        auto b        = migraphx::op::broadcast{1, {2, 2, 4, 5}};
        auto x        = mm2->add_parameter("x", s);
        auto y        = mm2->add_parameter("y", s);
        auto one      = mm2->add_literal(1);
        auto two      = mm2->add_literal(2);
        auto concat1  = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto concat2  = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concat2b = mm2->add_instruction(b, concat2);
        auto sum      = mm2->add_instruction(migraphx::make_op("add"), concat1, concat2b);
        auto relu     = mm2->add_instruction(migraphx::make_op("relu"), sum);
        mm2->add_instruction(pass_op{}, relu);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_concat_add_relu_broadcast_same_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {2, 1, 4, 5}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {2, 1, 4, 5}};
        auto x     = mm1->add_parameter("x", s);
        auto y     = mm1->add_parameter("y", s);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto concat =
            mm1->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), relu1, relu2);
        mm1->add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto b       = migraphx::op::broadcast{1, {2, 1, 4, 5}};
        auto x       = mm2->add_parameter("x", s);
        auto y       = mm2->add_parameter("y", s);
        auto one     = mm2->add_literal(1);
        auto oneb    = mm2->add_instruction(b, one);
        auto two     = mm2->add_literal(2);
        auto twob    = mm2->add_instruction(b, two);
        auto concat1 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto concat2 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), oneb, twob);
        auto sum     = mm2->add_instruction(migraphx::make_op("add"), concat1, concat2);
        auto relu    = mm2->add_instruction(migraphx::make_op("relu"), sum);
        mm2->add_instruction(pass_op{}, relu);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_div_const)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto two  = mm1->add_literal(2);
        mm1->add_instruction(migraphx::make_op("div"), x, two);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2  = p2.get_main_module();
        auto x     = mm2->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto two   = mm2->add_literal(2);
        auto recip = mm2->insert_instruction(std::next(two), migraphx::make_op("recip"), two);
        mm2->add_instruction(migraphx::make_op("mul"), x, recip);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_sub_const)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto two  = mm1->add_literal(2);
        mm1->add_instruction(migraphx::make_op("sub"), x, two);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto two  = mm2->add_literal(2);
        auto neg  = mm2->insert_instruction(std::next(two), migraphx::make_op("neg"), two);
        mm2->add_instruction(migraphx::make_op("add"), x, neg);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_rsqrt)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto sqrt = mm1->add_instruction(migraphx::make_op("sqrt"), x);
        mm1->add_instruction(migraphx::make_op("recip"), sqrt);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1}});
        mm2->add_instruction(migraphx::make_op("rsqrt"), x);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_rsqrt_multi_use)
{
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto x     = mm1->add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto sqrt  = mm1->add_instruction(migraphx::make_op("sqrt"), x);
        auto add   = mm1->add_instruction(migraphx::make_op("add"), sqrt, sqrt);
        auto rsqrt = mm1->add_instruction(migraphx::make_op("recip"), sqrt);
        mm1->add_instruction(migraphx::make_op("add"), rsqrt, add);
    }
    migraphx::program p2{p1};

    run_pass(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_slice_concat)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {256}};

    migraphx::program p1;
    {
        auto* mm1    = p1.get_main_module();
        auto x       = mm1->add_parameter("x", s);
        auto y       = mm1->add_parameter("y", s);
        auto xslice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {128}}}), x);
        auto xslice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {128}}, {"ends", {256}}}), x);
        auto yslice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {128}}}), y);
        auto yslice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {128}}, {"ends", {256}}}), y);
        auto concat = mm1->add_instruction(
            migraphx::make_op("concat", {{"axis", 0}}), xslice1, xslice2, yslice1, yslice2);
        mm1->add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2   = p2.get_main_module();
        auto x      = mm2->add_parameter("x", s);
        auto y      = mm2->add_parameter("y", s);
        auto concat = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        mm2->add_instruction(pass_op{}, concat);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(simplify_slice_concat_non_uniform)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {256}};

    migraphx::program p1;
    {
        auto* mm1    = p1.get_main_module();
        auto x       = mm1->add_parameter("x", s);
        auto y       = mm1->add_parameter("y", s);
        auto xslice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {64}}}), x);
        auto xslice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {192}}}), x);
        auto xslice3 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {192}}, {"ends", {256}}}), x);
        auto yslice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {64}}}), y);
        auto yslice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {192}}}), y);
        auto yslice3 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {192}}, {"ends", {256}}}), y);
        auto concat = mm1->add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                           xslice1,
                                           xslice2,
                                           xslice3,
                                           yslice1,
                                           yslice2,
                                           yslice3);
        mm1->add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2   = p2.get_main_module();
        auto x      = mm2->add_parameter("x", s);
        auto y      = mm2->add_parameter("y", s);
        auto concat = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        mm2->add_instruction(pass_op{}, concat);
    }

    EXPECT(p1 == p2);
}

TEST_CASE(simplify_slice_concat_flipped)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {256}};

    migraphx::program p1;
    {
        auto* mm1    = p1.get_main_module();
        auto x       = mm1->add_parameter("x", s);
        auto y       = mm1->add_parameter("y", s);
        auto xslice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {64}}}), x);
        auto xslice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {192}}, {"ends", {256}}}), x);
        auto xslice3 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {192}}}), x);
        auto yslice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {64}}}), y);
        auto yslice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {192}}, {"ends", {256}}}), y);
        auto yslice3 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {192}}}), y);
        auto concat = mm1->add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                           xslice1,
                                           xslice2,
                                           xslice3,
                                           yslice1,
                                           yslice2,
                                           yslice3);
        mm1->add_instruction(pass_op{}, concat);
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(simplify_split_add_relu)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 1, 4}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = mm1->add_instruction(migraphx::make_op("add"), relu1, relu2);
        mm1->add_instruction(pass_op{}, add);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto b       = migraphx::op::broadcast{1, {3, 2, 4}};
        auto input   = mm2->add_parameter("input", s);
        auto one     = mm2->add_literal(1);
        auto two     = mm2->add_literal(2);
        auto concat  = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = mm2->add_instruction(b, concat);
        auto sum     = mm2->add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = mm2->add_instruction(migraphx::make_op("relu"), sum);
        auto x       = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto y = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto add = mm2->add_instruction(migraphx::make_op("add"), x, y);
        mm2->add_instruction(pass_op{}, add);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_split_add_relu_reshape)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 1, 4}};
        auto r     = migraphx::op::reshape{{3, 4}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one      = mm1->add_literal(1);
        auto oneb     = mm1->add_instruction(b, one);
        auto two      = mm1->add_literal(2);
        auto twob     = mm1->add_instruction(b, two);
        auto sum1     = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1    = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto reshape1 = mm1->add_instruction(r, relu1);
        auto sum2     = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2    = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto reshape2 = mm1->add_instruction(r, relu2);
        auto add      = mm1->add_instruction(migraphx::make_op("add"), reshape1, reshape2);
        mm1->add_instruction(pass_op{}, add);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto b       = migraphx::op::broadcast{1, {3, 2, 4}};
        auto input   = mm2->add_parameter("input", s);
        auto one     = mm2->add_literal(1);
        auto two     = mm2->add_literal(2);
        auto concat  = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = mm2->add_instruction(b, concat);
        auto sum     = mm2->add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = mm2->add_instruction(migraphx::make_op("relu"), sum);
        auto rsp     = mm2->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 8}}}), relu);
        auto slc1    = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), rsp);
        auto slc2 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {8}}}), rsp);
        auto add = mm2->add_instruction(migraphx::make_op("add"), slc1, slc2);
        mm2->add_instruction(pass_op{}, add);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_slice_different_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4, 2}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto r     = migraphx::op::reshape{{3, 2, 4}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto one  = mm1->add_literal(1);
        auto oneb = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {3, 1, 4, 2}}}), one);
        auto two  = mm1->add_literal(2);
        auto twob = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 3}, {"dims", {3, 2, 4, 1}}}), two);
        auto sum1     = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1    = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto reshape1 = mm1->add_instruction(r, relu1);
        auto sum2     = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2    = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto reshape2 = mm1->add_instruction(r, relu2);
        auto add      = mm1->add_instruction(migraphx::make_op("add"), reshape1, reshape2);
        mm1->add_instruction(pass_op{}, add);
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_slice_missing_begining_slice)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 3, 4}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 1, 4}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {3}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = mm1->add_instruction(migraphx::make_op("add"), relu1, relu2);
        mm1->add_instruction(pass_op{}, add);
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_slice_missing_middle_slice)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 3, 4}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 1, 4}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {3}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = mm1->add_instruction(migraphx::make_op("add"), relu1, relu2);
        mm1->add_instruction(pass_op{}, add);
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_slice_missing_end_slice)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 3, 4}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 1, 4}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = mm1->add_instruction(migraphx::make_op("add"), relu1, relu2);
        mm1->add_instruction(pass_op{}, add);
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_split_add_relu_concat_same_axis)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 1, 4}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto concat =
            mm1->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), relu1, relu2);
        mm1->add_instruction(pass_op{}, concat);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto b       = migraphx::op::broadcast{1, {3, 2, 4}};
        auto input   = mm2->add_parameter("input", s);
        auto one     = mm2->add_literal(1);
        auto two     = mm2->add_literal(2);
        auto concat  = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = mm2->add_instruction(b, concat);
        auto sum     = mm2->add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = mm2->add_instruction(migraphx::make_op("relu"), sum);
        mm2->add_instruction(pass_op{}, relu);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_split_add_relu_multi_axes)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4, 6}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 1, 4, 3}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1, 3}}, {"starts", {0, 0}}, {"ends", {1, 3}}}),
            input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1, 3}}, {"starts", {1, 3}}, {"ends", {2, 6}}}),
            input);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto add   = mm1->add_instruction(migraphx::make_op("add"), relu1, relu2);
        mm1->add_instruction(pass_op{}, add);
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_split_add_relu_used_multiple_split1)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 1, 4}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto add1  = mm1->add_instruction(migraphx::make_op("add"), relu1, relu2);
        auto add2  = mm1->add_instruction(migraphx::make_op("add"), x, add1);
        mm1->add_instruction(pass_op{}, add2);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2  = p2.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 2, 4}};
        auto input = mm2->add_parameter("input", s);
        auto slice = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto one     = mm2->add_literal(1);
        auto two     = mm2->add_literal(2);
        auto concat  = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = mm2->add_instruction(b, concat);
        auto sum     = mm2->add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = mm2->add_instruction(migraphx::make_op("relu"), sum);
        auto x       = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto y = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto add1 = mm2->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm2->add_instruction(migraphx::make_op("add"), slice, add1);
        mm2->add_instruction(pass_op{}, add2);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_split_add_relu_used_multiple_split2)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 1, 4}};
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto z     = mm1->add_instruction(migraphx::make_op("relu"), x);
        auto one   = mm1->add_literal(1);
        auto oneb  = mm1->add_instruction(b, one);
        auto two   = mm1->add_literal(2);
        auto twob  = mm1->add_instruction(b, two);
        auto sum1  = mm1->add_instruction(migraphx::make_op("add"), x, oneb);
        auto relu1 = mm1->add_instruction(migraphx::make_op("relu"), sum1);
        auto sum2  = mm1->add_instruction(migraphx::make_op("add"), y, twob);
        auto relu2 = mm1->add_instruction(migraphx::make_op("relu"), sum2);
        auto add1  = mm1->add_instruction(migraphx::make_op("add"), relu1, relu2);
        auto add2  = mm1->add_instruction(migraphx::make_op("add"), z, add1);
        mm1->add_instruction(pass_op{}, add2);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2  = p2.get_main_module();
        auto b     = migraphx::op::broadcast{1, {3, 2, 4}};
        auto input = mm2->add_parameter("input", s);
        auto slice = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto z       = mm2->add_instruction(migraphx::make_op("relu"), slice);
        auto one     = mm2->add_literal(1);
        auto two     = mm2->add_literal(2);
        auto concat  = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), one, two);
        auto concatb = mm2->add_instruction(b, concat);
        auto sum     = mm2->add_instruction(migraphx::make_op("add"), input, concatb);
        auto relu    = mm2->add_instruction(migraphx::make_op("relu"), sum);
        auto x       = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto y = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto add1 = mm2->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm2->add_instruction(migraphx::make_op("add"), z, add1);
        mm2->add_instruction(pass_op{}, add2);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_split_between_add)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto input = mm1->add_parameter("input", s);
        auto x     = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto sum = mm1->add_instruction(migraphx::make_op("add"), x, y);
        mm1->add_instruction(pass_op{}, sum);
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_dot_horiz)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 2}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto input = mm1->add_parameter("input", s);
        auto a     = mm1->add_literal(migraphx::generate_literal(s, 0));
        auto b     = mm1->add_literal(migraphx::generate_literal(s, 1));
        auto x     = mm1->add_instruction(migraphx::make_op("dot"), input, a);
        auto y     = mm1->add_instruction(migraphx::make_op("dot"), input, b);
        auto sum   = mm1->add_instruction(migraphx::make_op("add"), x, y);
        mm1->add_instruction(pass_op{}, sum);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2   = p2.get_main_module();
        auto input  = mm2->add_parameter("input", s);
        auto a      = mm2->add_literal(migraphx::generate_literal(s, 0));
        auto b      = mm2->add_literal(migraphx::generate_literal(s, 1));
        auto concat = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), a, b);
        auto dot    = mm2->add_instruction(migraphx::make_op("dot"), input, concat);
        auto x      = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), dot);
        auto y = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {4}}}), dot);
        auto sum = mm2->add_instruction(migraphx::make_op("add"), x, y);
        mm2->add_instruction(pass_op{}, sum);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_dot_horiz_same_constant)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 2}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto input = mm1->add_parameter("input", s);
        auto a     = mm1->add_literal(migraphx::generate_literal(s, 0));
        auto x     = mm1->add_instruction(migraphx::make_op("dot"), input, a);
        auto y     = mm1->add_instruction(migraphx::make_op("dot"), input, a);
        auto sum   = mm1->add_instruction(migraphx::make_op("add"), x, y);
        mm1->add_instruction(pass_op{}, sum);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2   = p2.get_main_module();
        auto input  = mm2->add_parameter("input", s);
        auto a      = mm2->add_literal(migraphx::generate_literal(s, 0));
        auto concat = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), a, a);
        auto dot    = mm2->add_instruction(migraphx::make_op("dot"), input, concat);
        auto x      = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), dot);
        auto y = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {4}}}), dot);
        auto sum = mm2->add_instruction(migraphx::make_op("add"), x, y);
        mm2->add_instruction(pass_op{}, sum);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_dot_horiz_flipped)
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 2}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto input = mm1->add_parameter("input", s);
        auto a     = mm1->add_literal(migraphx::generate_literal(s, 0));
        auto b     = mm1->add_literal(migraphx::generate_literal(s, 1));
        auto x     = mm1->add_instruction(migraphx::make_op("dot"), input, a);
        auto y     = mm1->add_instruction(migraphx::make_op("dot"), b, input);
        auto sum   = mm1->add_instruction(migraphx::make_op("add"), x, y);
        mm1->add_instruction(pass_op{}, sum);
    }

    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_conv_horiz)
{
    auto s  = migraphx::shape{migraphx::shape::int32_type, {8, 3, 64, 64}};
    auto ws = migraphx::shape{migraphx::shape::int32_type, {12, 3, 3, 3}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto input = mm1->add_parameter("input", s);
        auto a     = mm1->add_literal(migraphx::generate_literal(ws, 0));
        auto b     = mm1->add_literal(migraphx::generate_literal(ws, 1));
        auto x     = mm1->add_instruction(migraphx::make_op("convolution"), input, a);
        auto y     = mm1->add_instruction(migraphx::make_op("convolution"), input, b);
        auto sum   = mm1->add_instruction(migraphx::make_op("add"), x, y);
        mm1->add_instruction(pass_op{}, sum);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2   = p2.get_main_module();
        auto input  = mm2->add_parameter("input", s);
        auto a      = mm2->add_literal(migraphx::generate_literal(ws, 0));
        auto b      = mm2->add_literal(migraphx::generate_literal(ws, 1));
        auto concat = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), a, b);
        auto conv   = mm2->add_instruction(migraphx::make_op("convolution"), input, concat);
        auto x      = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}), conv);
        auto y = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {12}}, {"ends", {24}}}), conv);
        auto sum = mm2->add_instruction(migraphx::make_op("add"), x, y);
        mm2->add_instruction(pass_op{}, sum);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_group_conv_horiz)
{
    auto s  = migraphx::shape{migraphx::shape::int32_type, {1, 32, 111, 111}};
    auto ws = migraphx::shape{migraphx::shape::int32_type, {32, 1, 7, 7}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto x     = mm1->add_parameter("x", s);
        auto w1    = mm1->add_literal(migraphx::generate_literal(ws, 1));
        auto w2    = mm1->add_literal(migraphx::generate_literal(ws, 2));
        auto conv1 = mm1->add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {3, 3}}, {"stride", {2, 2}}, {"dilation", {1, 1}}, {"group", 32}}),
            x,
            w1);
        auto conv2 = mm1->add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {3, 3}}, {"stride", {2, 2}}, {"dilation", {1, 1}}, {"group", 32}}),
            x,
            w2);
        mm1->add_instruction(pass_op{}, conv1, conv2);
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_conv_horiz_grouped)
{
    auto s   = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    auto ws1 = migraphx::shape{migraphx::shape::int32_type, {6, 6, 3, 3}};
    auto ws2 = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto input = mm1->add_parameter("input", s);
        auto a     = mm1->add_literal(migraphx::generate_literal(ws1, 0));
        auto b     = mm1->add_literal(migraphx::generate_literal(ws1, 1));
        auto c     = mm1->add_literal(migraphx::generate_literal(ws2, 2));
        auto d     = mm1->add_literal(migraphx::generate_literal(ws2, 3));
        auto convx =
            mm1->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, a);
        auto convy =
            mm1->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, b);
        auto dotx = mm1->add_instruction(migraphx::make_op("dot"), input, c);
        auto doty = mm1->add_instruction(migraphx::make_op("dot"), input, d);
        auto sum1 = mm1->add_instruction(migraphx::make_op("add"), convx, convy);
        auto sum2 = mm1->add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sum3 = mm1->add_instruction(migraphx::make_op("add"), sum1, sum2);

        mm1->add_instruction(pass_op{}, sum3);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto input   = mm2->add_parameter("input", s);
        auto a       = mm2->add_literal(migraphx::generate_literal(ws1, 0));
        auto b       = mm2->add_literal(migraphx::generate_literal(ws1, 1));
        auto c       = mm2->add_literal(migraphx::generate_literal(ws2, 2));
        auto d       = mm2->add_literal(migraphx::generate_literal(ws2, 3));
        auto concat1 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), a, b);
        auto concat2 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), c, d);
        auto conv    = mm2->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, concat1);
        auto convx = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {6}}}), conv);
        auto convy = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {6}}, {"ends", {12}}}), conv);
        auto sum1 = mm2->add_instruction(migraphx::make_op("add"), convx, convy);
        auto dot  = mm2->add_instruction(migraphx::make_op("dot"), input, concat2);
        auto dotx = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {64}}}), dot);
        auto doty = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {128}}}), dot);
        auto sum2 = mm2->add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sum3 = mm2->add_instruction(migraphx::make_op("add"), sum1, sum2);
        mm2->add_instruction(pass_op{}, sum3);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_conv_horiz_grouped_extra1)
{
    auto s   = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    auto ws1 = migraphx::shape{migraphx::shape::int32_type, {6, 6, 3, 3}};
    auto ws2 = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto input = mm1->add_parameter("input", s);
        auto a     = mm1->add_literal(migraphx::generate_literal(ws1, 0));
        auto b     = mm1->add_literal(migraphx::generate_literal(ws1, 1));
        auto c     = mm1->add_literal(migraphx::generate_literal(ws2, 2));
        auto d     = mm1->add_literal(migraphx::generate_literal(ws2, 3));
        auto e     = mm1->add_literal(migraphx::generate_literal(s, 4));
        auto convx =
            mm1->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, a);
        auto convy =
            mm1->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, b);
        auto dotx    = mm1->add_instruction(migraphx::make_op("dot"), input, c);
        auto doty    = mm1->add_instruction(migraphx::make_op("dot"), input, d);
        auto sqdiffx = mm1->add_instruction(migraphx::make_op("sqdiff"), input, e);
        auto sum1    = mm1->add_instruction(migraphx::make_op("add"), convx, convy);
        auto sum2    = mm1->add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sum3    = sqdiffx;
        auto sum4    = mm1->add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5    = mm1->add_instruction(migraphx::make_op("add"), sum4, sum3);
        mm1->add_instruction(pass_op{}, sum5);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto input   = mm2->add_parameter("input", s);
        auto a       = mm2->add_literal(migraphx::generate_literal(ws1, 0));
        auto b       = mm2->add_literal(migraphx::generate_literal(ws1, 1));
        auto c       = mm2->add_literal(migraphx::generate_literal(ws2, 2));
        auto d       = mm2->add_literal(migraphx::generate_literal(ws2, 3));
        auto e       = mm2->add_literal(migraphx::generate_literal(s, 4));
        auto concat1 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), a, b);
        auto concat2 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), c, d);
        auto conv    = mm2->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, concat1);
        auto convx = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {6}}}), conv);
        auto convy = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {6}}, {"ends", {12}}}), conv);
        auto sum1 = mm2->add_instruction(migraphx::make_op("add"), convx, convy);
        auto dot  = mm2->add_instruction(migraphx::make_op("dot"), input, concat2);
        auto dotx = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {64}}}), dot);
        auto doty = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {128}}}), dot);
        auto sum2    = mm2->add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sqdiffx = mm2->add_instruction(migraphx::make_op("sqdiff"), input, e);
        auto sum3    = sqdiffx;
        auto sum4    = mm2->add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5    = mm2->add_instruction(migraphx::make_op("add"), sum4, sum3);
        mm2->add_instruction(pass_op{}, sum5);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_conv_horiz_grouped_extra2)
{
    auto s   = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    auto ws1 = migraphx::shape{migraphx::shape::int32_type, {6, 6, 3, 3}};
    auto ws2 = migraphx::shape{migraphx::shape::int32_type, {8, 6, 64, 64}};
    migraphx::program p1;
    {
        auto* mm1  = p1.get_main_module();
        auto input = mm1->add_parameter("input", s);
        auto a     = mm1->add_literal(migraphx::generate_literal(ws1, 0));
        auto b     = mm1->add_literal(migraphx::generate_literal(ws1, 1));
        auto c     = mm1->add_literal(migraphx::generate_literal(ws2, 2));
        auto d     = mm1->add_literal(migraphx::generate_literal(ws2, 3));
        auto e     = mm1->add_literal(migraphx::generate_literal(s, 4));
        auto f     = mm1->add_literal(migraphx::generate_literal(s, 5));
        auto convx =
            mm1->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, a);
        auto convy =
            mm1->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, b);
        auto dotx    = mm1->add_instruction(migraphx::make_op("dot"), input, c);
        auto doty    = mm1->add_instruction(migraphx::make_op("dot"), input, d);
        auto sqdiffx = mm1->add_instruction(migraphx::make_op("sqdiff"), input, e);
        auto sqdiffy = mm1->add_instruction(migraphx::make_op("sqdiff"), input, f);
        auto sum1    = mm1->add_instruction(migraphx::make_op("add"), convx, convy);
        auto sum2    = mm1->add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sum3    = mm1->add_instruction(migraphx::make_op("add"), sqdiffx, sqdiffy);
        auto sum4    = mm1->add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5    = mm1->add_instruction(migraphx::make_op("add"), sum4, sum3);
        mm1->add_instruction(pass_op{}, sum5);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2    = p2.get_main_module();
        auto input   = mm2->add_parameter("input", s);
        auto a       = mm2->add_literal(migraphx::generate_literal(ws1, 0));
        auto b       = mm2->add_literal(migraphx::generate_literal(ws1, 1));
        auto c       = mm2->add_literal(migraphx::generate_literal(ws2, 2));
        auto d       = mm2->add_literal(migraphx::generate_literal(ws2, 3));
        auto e       = mm2->add_literal(migraphx::generate_literal(s, 4));
        auto f       = mm2->add_literal(migraphx::generate_literal(s, 5));
        auto concat1 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), a, b);
        auto concat2 = mm2->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), c, d);
        auto conv    = mm2->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}}), input, concat1);
        auto convx = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {6}}}), conv);
        auto convy = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {6}}, {"ends", {12}}}), conv);
        auto sum1 = mm2->add_instruction(migraphx::make_op("add"), convx, convy);
        auto dot  = mm2->add_instruction(migraphx::make_op("dot"), input, concat2);
        auto dotx = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {64}}}), dot);
        auto doty = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {128}}}), dot);
        auto sum2    = mm2->add_instruction(migraphx::make_op("add"), dotx, doty);
        auto sqdiffx = mm2->add_instruction(migraphx::make_op("sqdiff"), input, e);
        auto sqdiffy = mm2->add_instruction(migraphx::make_op("sqdiff"), input, f);
        auto sum3    = mm2->add_instruction(migraphx::make_op("add"), sqdiffx, sqdiffy);
        auto sum4    = mm2->add_instruction(migraphx::make_op("add"), sum1, sum2);
        auto sum5    = mm2->add_instruction(migraphx::make_op("add"), sum4, sum3);
        mm2->add_instruction(pass_op{}, sum5);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(simplify_mul_slice_conv_horiz_fusion)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto x    = mm1->add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w    = mm1->add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto conv   = mm1->add_instruction(migraphx::make_op("convolution"), x, w);
        auto slice1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), conv);
        auto a1 =
            mm1->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 1));
        auto b1 = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 384, 17, 17}}}), a1);
        auto mul = mm1->add_instruction(migraphx::make_op("mul"), slice1, b1);
        auto a2 =
            mm1->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 2));
        auto b2 = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 384, 17, 17}}}), a2);
        auto add1 = mm1->add_instruction(migraphx::make_op("add"), mul, b2);
        auto a3 =
            mm1->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 3));
        auto b3 = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 384, 17, 17}}}), a3);
        auto slice2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {384}}, {"ends", {768}}}), conv);
        auto add2 = mm1->add_instruction(migraphx::make_op("add"), slice2, b3);
        mm1->add_instruction(pass_op{}, add1, add2);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto x    = mm2->add_parameter("x", {migraphx::shape::int32_type, {1, 1024, 17, 17}});
        auto w    = mm2->add_literal(
            migraphx::generate_literal({migraphx::shape::int32_type, {768, 1024, 1, 1}}));
        auto wslice1 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {384}}}), w);
        auto a1 =
            mm2->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 1));
        auto b1 = mm2->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"dims", {384, 1024, 1, 1}}}), a1);
        auto mul     = mm2->add_instruction(migraphx::make_op("mul"), b1, wslice1);
        auto wslice2 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {384}}, {"ends", {768}}}), w);
        auto concat1 =
            mm2->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), mul, wslice2);
        auto conv = mm2->add_instruction(migraphx::make_op("convolution"), x, concat1);
        auto a2 =
            mm2->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 2));
        auto a3 =
            mm2->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {384}}, 3));
        auto concat2 = mm2->add_instruction(migraphx::make_op("concat"), a2, a3);
        auto b4      = mm2->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 768, 17, 17}}}), concat2);
        auto add    = mm2->add_instruction(migraphx::make_op("add"), conv, b4);
        auto slice1 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {384}}}), add);
        auto slice2 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {384}}, {"ends", {768}}}), add);
        mm2->add_instruction(pass_op{}, slice1, slice2);
    }
    EXPECT(p1.sort() == p2.sort());
}
TEST_CASE(reorder_reshape_slice)
{
    std::vector<int64_t> perm0 = {0, 2, 1, 3};
    std::vector<int64_t> perm1 = {0, 2, 3, 1};
    auto create_p1             = [&](std::size_t batch_size) {
        migraphx::program p1;
        auto* mm1  = p1.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {batch_size, 128, 1920}};
        auto input = mm1->add_parameter("input", s);
        auto slc0  = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {640}}}), input);
        auto slc1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {640}}, {"ends", {1280}}}),
            input);
        auto slc2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1280}}, {"ends", {1920}}}),
            input);

        auto c0 = mm1->add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = mm1->add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = mm1->add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {static_cast<int64_t>(batch_size), 128, 10, 64};
        auto r0 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto t0 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), r0);
        auto t1 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), r1);
        auto t2 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm1}}), r2);

        auto sum = mm1->add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = mm1->add_instruction(migraphx::make_op("dot"), sum, t2);
        mm1->add_return({ret});

        return p1;
    };

    auto create_p2 = [&](std::size_t batch_size) {
        migraphx::program p2;
        auto* mm2  = p2.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {batch_size, 128, 1920}};
        auto input = mm2->add_parameter("input", s);
        std::vector<int64_t> lens = {static_cast<int64_t>(batch_size), 128, 30, 64};
        auto r = mm2->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), input);

        auto slc0 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {10}}}), r);
        auto slc1 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {10}}, {"ends", {20}}}), r);
        auto slc2 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {20}}, {"ends", {30}}}), r);

        auto t0 = mm2->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), slc0);
        auto t1 = mm2->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), slc1);
        auto t2 = mm2->add_instruction(migraphx::make_op("transpose", {{"dims", perm1}}), slc2);

        auto sum = mm2->add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = mm2->add_instruction(migraphx::make_op("dot"), sum, t2);
        mm2->add_return({ret});

        return p2;
    };

    auto test = [&](std::size_t batch_size) {
        auto p1 = create_p1(batch_size);
        run_pass(p1);
        auto p2 = create_p2(batch_size);
        EXPECT(p1.sort() == p2.sort());
    };

    test(1);
    test(4);
    test(8);
}

TEST_CASE(reorder_reshape_slice_move_axis1)
{
    auto create_p1 = [](std::size_t batch_size) {
        migraphx::program p1;
        auto* mm1 = p1.get_main_module();
        auto s    = migraphx::shape{migraphx::shape::float_type, {batch_size, 256, 96}};
        std::vector<int64_t> perm0 = {0, 2, 1, 3};
        std::vector<int64_t> perm1 = {0, 2, 3, 1};
        auto input                 = mm1->add_parameter("input", s);
        auto slc0                  = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {32}}}), input);
        auto slc1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {32}}, {"ends", {64}}}), input);
        auto slc2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {64}}, {"ends", {96}}}), input);

        auto c0 = mm1->add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = mm1->add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = mm1->add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {static_cast<int64_t>(batch_size), 64, 4, 32};
        auto r0 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto t0 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), r0);
        auto t1 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), r1);
        auto t2 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm1}}), r2);

        auto sum = mm1->add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = mm1->add_instruction(migraphx::make_op("dot"), sum, t2);
        mm1->add_return({ret});

        return p1;
    };

    auto create_p2 = [](std::size_t batch_size) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto s   = migraphx::shape{migraphx::shape::float_type, {batch_size, 256, 96}};
        std::vector<int64_t> perm0 = {0, 2, 1, 3};
        std::vector<int64_t> perm1 = {0, 2, 3, 1};
        auto input                 = mm->add_parameter("input", s);
        std::vector<int64_t> lens  = {static_cast<int64_t>(batch_size), 64, 4, 96};
        auto rsp  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), input);
        auto slc0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {32}}}), rsp);
        auto t0   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), slc0);
        auto slc1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {32}}, {"ends", {64}}}), rsp);
        auto t1   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), slc1);
        auto slc2 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {96}}}), rsp);
        auto t2 = mm->add_instruction(migraphx::make_op("transpose", {{"dims", perm1}}), slc2);

        auto sum = mm->add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = mm->add_instruction(migraphx::make_op("dot"), sum, t2);
        mm->add_return({ret});

        return p;
    };

    auto test = [&](std::size_t batch_size) {
        auto p1 = create_p1(batch_size);
        auto p2 = create_p2(batch_size);
        run_pass(p1);
        EXPECT(p1.sort() == p2.sort());
    };

    test(4);
    test(8);
}

TEST_CASE(reorder_reshape_slice_move_axis2)
{
    auto create_p1 = [] {
        migraphx::program p1;
        auto* mm1 = p1.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {128, 96}};
        auto input = mm1->add_parameter("input", s);
        auto slc0  = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {32}}}), input);
        auto slc1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {32}}, {"ends", {64}}}), input);
        auto slc2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {64}}, {"ends", {96}}}), input);

        auto c0 = mm1->add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = mm1->add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = mm1->add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {1, 16, 8, 32};
        auto r0 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto sum = mm1->add_instruction(migraphx::make_op("add"), r0, r1);
        auto ret = mm1->add_instruction(migraphx::make_op("mul"), sum, r2);
        mm1->add_return({ret});

        return p1;
    };

    auto create_p2 = [] {
        migraphx::program p;
        auto* mm                  = p.get_main_module();
        auto s                    = migraphx::shape{migraphx::shape::float_type, {128, 96}};
        auto input                = mm->add_parameter("input", s);
        std::vector<int64_t> lens = {1, 16, 8, 96};
        auto rsp  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), input);
        auto slc0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {32}}}), rsp);
        auto slc1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {32}}, {"ends", {64}}}), rsp);
        auto slc2 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {64}}, {"ends", {96}}}), rsp);

        auto sum = mm->add_instruction(migraphx::make_op("add"), slc0, slc1);
        auto ret = mm->add_instruction(migraphx::make_op("mul"), sum, slc2);
        mm->add_return({ret});

        return p;
    };

    auto p1 = create_p1();
    auto p2 = create_p2();
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reorder_reshape_slice_not_apply)
{
    auto create_p = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {128, 96}};
        auto input = mm->add_parameter("input", s);
        auto slc0  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {32}}}), input);
        auto slc1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {32}}, {"ends", {64}}}), input);
        auto slc2 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {64}}, {"ends", {96}}}), input);

        auto c0 = mm->add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = mm->add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = mm->add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens = {1, 16, 16, 16};
        auto r0 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c2);

        auto sum = mm->add_instruction(migraphx::make_op("add"), r0, r1);
        auto ret = mm->add_instruction(migraphx::make_op("mul"), sum, r2);
        mm->add_return({ret});

        return p;
    };

    auto p1 = create_p();
    auto p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reorder_reshape_slice_diff_dims)
{
    auto create_p1 = [](std::size_t batch_size) {
        migraphx::program p1;
        auto* mm1 = p1.get_main_module();
        auto s    = migraphx::shape{migraphx::shape::float_type, {batch_size, 96, 96}};
        std::vector<int64_t> perm0 = {0, 2, 1, 3};
        std::vector<int64_t> perm1 = {0, 2, 3, 1};
        auto input                 = mm1->add_parameter("input", s);
        auto slc0                  = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {32}}}), input);
        auto slc1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {32}}, {"ends", {64}}}), input);
        auto slc2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {64}}, {"ends", {96}}}), input);

        auto c0 = mm1->add_instruction(migraphx::make_op("contiguous"), slc0);
        auto c1 = mm1->add_instruction(migraphx::make_op("contiguous"), slc1);
        auto c2 = mm1->add_instruction(migraphx::make_op("contiguous"), slc2);

        std::vector<int64_t> lens  = {static_cast<int64_t>(batch_size), 32, 3, 32};
        std::vector<int64_t> lens1 = {static_cast<int64_t>(batch_size), 48, 2, 32};
        auto r0 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c0);
        auto r1 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), c1);
        auto r2 = mm1->add_instruction(migraphx::make_op("reshape", {{"dims", lens1}}), c2);

        mm1->add_return({r0, r1, r2});

        return p1;
    };

    auto test = [&](std::size_t batch_size) {
        auto p1 = create_p1(batch_size);
        auto p2 = p1;
        run_pass(p1);
        EXPECT(p1.sort() == p2.sort());
    };

    test(4);
    test(8);
}

TEST_CASE(reorder_slice_trans)
{
    std::vector<int64_t> perm = {0, 2, 1};
    auto create_p1            = [&](std::size_t batch_size) {
        migraphx::program p1;
        auto* mm1  = p1.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {batch_size, 128, 1920}};
        auto input = mm1->add_parameter("input", s);
        auto slc0  = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {640}}}), input);
        auto slc1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {640}}, {"ends", {1280}}}),
            input);
        auto slc2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1280}}, {"ends", {1920}}}),
            input);

        auto t0 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm}}), slc0);
        auto t1 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm}}), slc1);
        auto t2 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm}}), slc2);

        auto sum = mm1->add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = mm1->add_instruction(migraphx::make_op("mul"), sum, t2);
        mm1->add_return({ret});

        return p1;
    };

    auto create_p2 = [&](std::size_t batch_size) {
        migraphx::program p2;
        auto* mm2  = p2.get_main_module();
        auto s     = migraphx::shape{migraphx::shape::float_type, {batch_size, 128, 1920}};
        auto input = mm2->add_parameter("input", s);
        auto r     = mm2->add_instruction(migraphx::make_op("transpose", {{"dims", perm}}), input);

        auto slc0 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {640}}}), r);
        auto slc1 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {640}}, {"ends", {1280}}}), r);
        auto slc2 = mm2->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1280}}, {"ends", {1920}}}), r);

        auto sum = mm2->add_instruction(migraphx::make_op("add"), slc0, slc1);
        auto ret = mm2->add_instruction(migraphx::make_op("mul"), sum, slc2);
        mm2->add_return({ret});

        return p2;
    };

    auto test = [&](std::size_t batch_size) {
        auto p1 = create_p1(batch_size);
        run_pass(p1);
        auto p2 = create_p2(batch_size);
        EXPECT(p1.sort() == p2.sort());
    };

    test(1);
    test(8);
}

TEST_CASE(reorder_slice_trans_diff_perm)
{
    auto create_p1 = [](std::size_t batch_size) {
        migraphx::program p1;
        auto* mm1 = p1.get_main_module();
        auto s    = migraphx::shape{migraphx::shape::float_type, {batch_size, 128, 1920}};
        std::vector<int64_t> perm0 = {0, 2, 1};
        std::vector<int64_t> perm1 = {0, 1, 2};
        auto input                 = mm1->add_parameter("input", s);
        auto slc0                  = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {640}}}), input);
        auto slc1 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {640}}, {"ends", {1280}}}),
            input);
        auto slc2 = mm1->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1280}}, {"ends", {1920}}}),
            input);

        auto t0 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), slc0);
        auto t1 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm0}}), slc1);
        auto t2 = mm1->add_instruction(migraphx::make_op("transpose", {{"dims", perm1}}), slc2);

        auto sum = mm1->add_instruction(migraphx::make_op("add"), t0, t1);
        auto ret = mm1->add_instruction(migraphx::make_op("dot"), sum, t2);
        mm1->add_return({ret});

        return p1;
    };

    auto test = [&](std::size_t batch_size) {
        auto p1 = create_p1(batch_size);
        run_pass(p1);
        auto p2 = p1;
        EXPECT(p1.sort() == p2.sort());
    };

    test(1);
    test(4);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
