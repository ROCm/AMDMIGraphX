#include <migraphx/optimize_qdq_format.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>

#include <migraphx/serialize.hpp>

#include <migraphx/verify.hpp>

bool is_convolution(migraphx::instruction& ins) { return ins.name() == "convolution"; }
bool is_dot(migraphx::instruction& ins) { return ins.name() == "dot"; }

struct match_find_qdq
{
    bool match_found = false;

    auto matcher() const
    {
        return migraphx::match::name("dequantizelinear")(
            migraphx::match::arg(0)(migraphx::match::name("quantizelinear")));
    }
    void apply(migraphx::module&, migraphx::match::matcher_result) { match_found = true; }
};

TEST_CASE(optimize_qdq)
{
    migraphx::shape s1{migraphx::shape::int8_type, {1}};
    migraphx::shape s2{migraphx::shape::int8_type, {1280, 1000}};
    migraphx::shape s3{migraphx::shape::int8_type, {1000}};
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s5{migraphx::shape::int32_type, {1}};
    migraphx::shape s6{migraphx::shape::int32_type, {1280}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};
    migraphx::shape s8{migraphx::shape::float_type, {1}};

    std::vector<std::vector<float>> floats{{0.00122128},
                                           {0.168607},
                                           {0.00260497},
                                           {0.0235534},
                                           {0.0235534},
                                           {0.023622},
                                           {0.0162265},
                                           {0.0138147}};
    std::vector<std::vector<int>> ints{{-35}, {-127}, {-18}, {0}};

    auto create_program = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto p1  = mm->add_parameter("p1", s2);
        auto p2  = mm->add_parameter("p2", s3);
        auto p3  = mm->add_parameter("p3", s4);
        auto p4  = mm->add_parameter("p4", s6);
        auto pa  = mm->add_parameter("pa", s7);

        auto l1  = mm->add_literal(s1, ints[0]);
        auto l2  = mm->add_literal(s1, ints[1]);
        auto l3  = mm->add_literal(s1, ints[1]);
        auto l4  = mm->add_literal(s1, ints[1]);
        auto l5  = mm->add_literal(s1, ints[2]);
        auto l6  = mm->add_literal(s5, ints[3]);
        auto l7  = mm->add_literal(s1, ints[3]);
        auto l8  = mm->add_literal(s1, ints[3]);
        auto l9  = mm->add_literal(s8, floats[0]);
        auto l10 = mm->add_literal(s8, floats[1]);
        auto l11 = mm->add_literal(s8, floats[2]);
        auto l12 = mm->add_literal(s8, floats[3]);
        auto l13 = mm->add_literal(s8, floats[4]);
        auto l14 = mm->add_literal(s8, floats[5]);
        auto l15 = mm->add_literal(s8, floats[6]);
        auto l16 = mm->add_literal(s8, floats[7]);

        auto mb1 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1280, 320, 1, 1}}}), l16);
        auto mb2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1280, 320, 1, 1}}}), l7);
        auto d1  = mm->add_instruction(migraphx::make_op("dequantizelinear"), p3, mb1, mb2);
        auto mb3 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1280}}}), l11);
        auto mb4 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {1280}}}), l6);
        auto d2 = mm->add_instruction(migraphx::make_op("dequantizelinear"), p4, mb3, mb4);
        auto mb5 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {1000}}}), l9);
        auto mb6 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", {1000}}}), l7);
        auto d3  = mm->add_instruction(migraphx::make_op("dequantizelinear"), p2, mb5, mb6);
        auto mb7 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1280, 1000}}}), l11);
        auto mb8 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1280, 1000}}}), l8);
        auto d4  = mm->add_instruction(migraphx::make_op("dequantizelinear"), p1, mb7, mb8);
        auto mb9 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 320, 7, 7}}}), l15);
        auto mb10 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 320, 7, 7}}}), l5);
        auto q1   = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, mb9, mb10);
        auto mb11 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 320, 7, 7}}}), l15);
        auto mb12 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 320, 7, 7}}}), l5);
        auto d5 = mm->add_instruction(migraphx::make_op("dequantizelinear"), q1, mb11, mb12);
        auto c1 = mm->add_instruction(migraphx::make_op("convolution",
                                                        {{"padding", {0, 0, 0, 0}},
                                                         {"stride", {1, 1}},
                                                         {"dilation", {1, 1}},
                                                         {"group", 1},
                                                         {"padding_mode", 0}}),
                                      d5,
                                      d1);
        auto b1 = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 1280, 7, 7}}}), d2);
        auto a1   = mm->add_instruction(migraphx::make_op("add"), c1, b1);
        auto mb13 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280, 7, 7}}}), l14);
        auto mb14 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280, 7, 7}}}), l4);
        auto q2   = mm->add_instruction(migraphx::make_op("quantizelinear"), a1, mb13, mb14);
        auto mb15 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280, 7, 7}}}), l14);
        auto mb16 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280, 7, 7}}}), l4);
        auto d6   = mm->add_instruction(migraphx::make_op("dequantizelinear"), q2, mb15, mb16);
        auto ap   = mm->add_instruction(migraphx::make_op("pooling",
                                                        {{"mode", "average"},
                                                         {"padding", {0, 0, 0, 0}},
                                                         {"stride", {1, 1}},
                                                         {"lengths", {7, 7}},
                                                         {"ceil_mode", 0}}),
                                      d6);
        auto mb17 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280, 1, 1}}}), l13);
        auto mb18 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280, 1, 1}}}), l3);
        auto q3   = mm->add_instruction(migraphx::make_op("quantizelinear"), ap, mb17, mb18);
        auto mb19 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280, 1, 1}}}), l13);
        auto mb20 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280, 1, 1}}}), l3);
        auto d7   = mm->add_instruction(migraphx::make_op("dequantizelinear"), q3, mb19, mb20);
        auto rs   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1}}}), d7);
        auto mb21 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280}}}), l12);
        auto mb22 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280}}}), l2);
        auto q4   = mm->add_instruction(migraphx::make_op("quantizelinear"), rs, mb21, mb22);
        auto mb23 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280}}}), l12);
        auto mb24 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1280}}}), l2);
        auto d8 = mm->add_instruction(migraphx::make_op("dequantizelinear"), q4, mb23, mb24);
        auto dot =
            mm->add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d8, d4);
        auto mb25 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1000}}}), l10);
        auto mb26 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1000}}}), l1);
        auto q5   = mm->add_instruction(migraphx::make_op("quantizelinear"), dot, mb25, mb26);
        auto mb27 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1000}}}), l10);
        auto mb28 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1000}}}), l1);
        auto d9   = mm->add_instruction(migraphx::make_op("dequantizelinear"), q5, mb27, mb28);
        auto mb29 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1000}}}), d3);
        auto a2 = mm->add_instruction(migraphx::make_op("add"), d9, mb29);
        mm->add_return({a2});

        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    migraphx::optimize_qdq_format opt;
    opt.apply(*p2.get_main_module());

    match_find_qdq m1, m2;
    migraphx::match::find_matches(*p1.get_main_module(), m1);
    migraphx::match::find_matches(*p2.get_main_module(), m2);

    EXPECT(m1.match_found and not m2.match_found);
    EXPECT(any_of(*p1.get_main_module(), &is_convolution));
    EXPECT(none_of(*p2.get_main_module(), &is_convolution));
    EXPECT(any_of(*p1.get_main_module(), &is_dot));
    EXPECT(none_of(*p2.get_main_module(), &is_dot));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
