#include <migraphx/simplify_qdq.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/instruction.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>

bool is_convolution(const migraphx::instruction& ins) { return ins.name() == "convolution"; }
bool is_dot(const migraphx::instruction& ins) { return ins.name() == "dot"; }

void run_pass(migraphx::module& m)
{
    migraphx::simplify_qdq sqdq;
    sqdq.apply(m);
}

migraphx::instruction_ref add_quantize_op(migraphx::module& m,
                                          const std::string& name,
                                          migraphx::instruction_ref x,
                                          migraphx::instruction_ref scale,
                                          migraphx::instruction_ref shift)
{
    auto lens = x->get_shape().lens();
    auto scale_mb =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", lens}}), scale);
    auto shift_mb =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"output_lens", lens}}), shift);
    return m.add_instruction(migraphx::make_op(name), x, scale_mb, shift_mb);
}

TEST_CASE(dot)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};
    migraphx::shape sh3{migraphx::shape::int8_type, {1000, 1024}};
    migraphx::shape sh4{migraphx::shape::float_type, {1280, 1024}};
    migraphx::shape sh5{migraphx::shape::int8_type, {1280, 1024}};
    migraphx::shape sh6{migraphx::shape::int8_type, {1280, 1000}};

    migraphx::module m1;
    {
        auto t1  = m1.add_parameter("t1", sh1);
        auto t2  = m1.add_parameter("t2", sh2);
        auto sc1 = m1.add_parameter("sc1", sh1);
        auto sc2 = m1.add_parameter("sc2", sh2);
        auto sc3 = m1.add_parameter("sc3", sh4);
        auto z1  = m1.add_parameter("z1", sh6);
        auto z2  = m1.add_parameter("z2", sh3);
        auto z3  = m1.add_parameter("z3", sh5);

        auto q1 = m1.add_instruction(migraphx::make_op("quantizelinear"), t1, sc1, z1);
        auto d1 = m1.add_instruction(migraphx::make_op("dequantizelinear"), q1, sc1, z1);
        auto q2 = m1.add_instruction(migraphx::make_op("quantizelinear"), t2, sc2, z2);
        auto d2 = m1.add_instruction(migraphx::make_op("dequantizelinear"), q2, sc2, z2);
        auto dot =
            m1.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d1, d2);
        auto q3 = m1.add_instruction(migraphx::make_op("quantizelinear"), dot, sc3, z3);
        auto d3 = m1.add_instruction(migraphx::make_op("dequantizelinear"), q3, sc3, z3);
        m1.add_return({d3});
    }

    migraphx::module m2;
    {
        auto t1  = m2.add_parameter("t1", sh1);
        auto t2  = m2.add_parameter("t2", sh2);
        auto sc1 = m2.add_parameter("sc1", sh1);
        auto sc2 = m2.add_parameter("sc2", sh2);
        auto sc3 = m2.add_parameter("sc3", sh4);
        auto z1  = m2.add_parameter("z1", sh6);
        auto z2  = m2.add_parameter("z2", sh3);
        auto z3  = m2.add_parameter("z3", sh5);

        auto q1 = m2.add_instruction(migraphx::make_op("quantizelinear"), t1, sc1, z1);
        auto q2 = m2.add_instruction(migraphx::make_op("quantizelinear"), t2, sc2, z2);
        auto dot =
            m2.add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), q1, q2);
        auto d3 = m2.add_instruction(migraphx::make_op("dequantizelinear"), dot, sc3, z3);
        m2.add_return({d3});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_add)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};
    migraphx::shape sh3{migraphx::shape::int8_type, {1000, 1024}};
    migraphx::shape sh4{migraphx::shape::float_type, {1280, 1024}};
    migraphx::shape sh5{migraphx::shape::int8_type, {1280, 1024}};
    migraphx::shape sh6{migraphx::shape::int8_type, {1280, 1000}};

    migraphx::module m1;
    {
        auto t1  = m1.add_parameter("t1", sh1);
        auto t2  = m1.add_parameter("t2", sh2);
        auto sc1 = m1.add_parameter("sc1", sh1);
        auto sc2 = m1.add_parameter("sc2", sh2);
        auto sc3 = m1.add_parameter("sc3", sh4);
        auto z1  = m1.add_parameter("z1", sh6);
        auto z2  = m1.add_parameter("z2", sh3);
        auto z3  = m1.add_parameter("z3", sh5);

        auto q1 = m1.add_instruction(migraphx::make_op("quantizelinear"), t1, sc1, z1);
        auto d1 = m1.add_instruction(migraphx::make_op("dequantizelinear"), q1, sc1, z1);
        auto q2 = m1.add_instruction(migraphx::make_op("quantizelinear"), t2, sc2, z2);
        auto d2 = m1.add_instruction(migraphx::make_op("dequantizelinear"), q2, sc2, z2);
        auto dot =
            m1.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d1, d2);
        auto q3  = m1.add_instruction(migraphx::make_op("quantizelinear"), dot, sc3, z3);
        auto d3  = m1.add_instruction(migraphx::make_op("dequantizelinear"), q3, sc3, z3);
        auto add = m1.add_instruction(migraphx::make_op("add"), d3, sc3);
        auto q4  = m1.add_instruction(migraphx::make_op("quantizelinear"), add, sc3, z3);
        auto d4  = m1.add_instruction(migraphx::make_op("dequantizelinear"), q4, sc3, z3);
        auto id  = m1.add_instruction(migraphx::make_op("identity"), d4);
        m1.add_return({id});
    }

    migraphx::module m2;
    {
        auto t1  = m2.add_parameter("t1", sh1);
        auto t2  = m2.add_parameter("t2", sh2);
        auto sc1 = m2.add_parameter("sc1", sh1);
        auto sc2 = m2.add_parameter("sc2", sh2);
        auto sc3 = m2.add_parameter("sc3", sh4);
        auto z1  = m2.add_parameter("z1", sh6);
        auto z2  = m2.add_parameter("z2", sh3);
        auto z3  = m2.add_parameter("z3", sh5);

        auto q1 = m2.add_instruction(migraphx::make_op("quantizelinear"), t1, sc1, z1);
        auto q2 = m2.add_instruction(migraphx::make_op("quantizelinear"), t2, sc2, z2);
        auto dot =
            m2.add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), q1, q2);
        auto d3  = m2.add_instruction(migraphx::make_op("dequantizelinear"), dot, sc3, z3);
        auto add = m2.add_instruction(migraphx::make_op("add"), d3, sc3);
        auto id  = m2.add_instruction(migraphx::make_op("identity"), add);
        m2.add_return({id});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(conv)
{
    migraphx::shape s1{migraphx::shape::int8_type, {1}};
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};
    migraphx::shape s8{migraphx::shape::float_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{0};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s7); // conv input (x)
        auto w   = m1.add_parameter("w", s4); // conv weights (w)
        auto l4  = m1.add_literal(s1, zero);
        auto l5  = m1.add_literal(s1, zero);
        auto l7  = m1.add_literal(s1, zero);
        auto l14 = m1.add_literal(s8, scale);
        auto l15 = m1.add_literal(s8, scale);
        auto l16 = m1.add_literal(s8, scale);

        auto d1 = add_quantize_op(m1, "dequantizelinear", w, l16, l7);
        auto q1 = add_quantize_op(m1, "quantizelinear", x, l15, l5);
        auto d5 = add_quantize_op(m1, "dequantizelinear", q1, l15, l5);
        auto c1 = m1.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     d5,
                                     d1);
        auto q2 = add_quantize_op(m1, "quantizelinear", c1, l14, l4);
        auto d6 = add_quantize_op(m1, "dequantizelinear", q2, l14, l4);
        m1.add_return({d6});
    }

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s7); // conv input (x)
        auto w   = m2.add_parameter("w", s4); // conv weights (w)
        auto l4  = m2.add_literal(s1, zero);
        auto l5  = m2.add_literal(s1, zero);
        auto l14 = m2.add_literal(s8, scale);
        auto l15 = m2.add_literal(s8, scale);

        auto q1 = add_quantize_op(m2, "quantizelinear", x, l15, l5);
        auto c1 = m2.add_instruction(migraphx::make_op("quant_convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     q1,
                                     w);
        auto d6 = add_quantize_op(m2, "dequantizelinear", c1, l14, l4);
        m2.add_return({d6});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(conv_bias_add)
{
    migraphx::shape s1{migraphx::shape::int8_type, {1}};
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s5{migraphx::shape::int32_type, {1}};
    migraphx::shape s6{migraphx::shape::int32_type, {1280}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};
    migraphx::shape s8{migraphx::shape::float_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{0};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s7); // conv input (x)
        auto w    = m1.add_parameter("w", s4); // conv weights (w)
        auto bias = m1.add_parameter("bias", s6);
        auto l4   = m1.add_literal(s1, zero);
        auto l5   = m1.add_literal(s1, zero);
        auto l6   = m1.add_literal(s5, zero);
        auto l7   = m1.add_literal(s1, zero);
        auto l11  = m1.add_literal(s8, scale);
        auto l14  = m1.add_literal(s8, scale);
        auto l15  = m1.add_literal(s8, scale);
        auto l16  = m1.add_literal(s8, scale);

        auto d1 = add_quantize_op(m1, "dequantizelinear", w, l16, l7);
        auto d2 = add_quantize_op(m1, "dequantizelinear", bias, l11, l6);
        auto q1 = add_quantize_op(m1, "quantizelinear", x, l15, l5);
        auto d5 = add_quantize_op(m1, "dequantizelinear", q1, l15, l5);
        auto c1 = m1.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     d5,
                                     d1);
        auto b1 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 1280, 7, 7}}}), d2);
        auto a1 = m1.add_instruction(migraphx::make_op("add"), c1, b1);
        auto q2 = add_quantize_op(m1, "quantizelinear", a1, l14, l4);
        auto d6 = add_quantize_op(m1, "dequantizelinear", q2, l14, l4);
        auto id = m1.add_instruction(migraphx::make_op("identity"), d6);
        m1.add_return({id});
    }

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s7); // conv input (x)
        auto w    = m2.add_parameter("w", s4); // conv weights (w)
        auto bias = m2.add_parameter("bias", s6);
        auto l4   = m2.add_literal(s1, zero);
        auto l5   = m2.add_literal(s1, zero);
        auto l6   = m2.add_literal(s5, zero);
        auto l11  = m2.add_literal(s8, scale);
        auto l14  = m2.add_literal(s8, scale);
        auto l15  = m2.add_literal(s8, scale);

        auto d2 = add_quantize_op(m2, "dequantizelinear", bias, l11, l6);
        auto q1 = add_quantize_op(m2, "quantizelinear", x, l15, l5);
        auto c1 = m2.add_instruction(migraphx::make_op("quant_convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     q1,
                                     w);
        auto b1 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 1280, 7, 7}}}), d2);
        auto d6 = add_quantize_op(m2, "dequantizelinear", c1, l14, l4);
        auto a1 = m2.add_instruction(migraphx::make_op("add"), d6, b1);
        auto id = m2.add_instruction(migraphx::make_op("identity"), a1);
        m2.add_return({id});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(mobilenet_snippet)
{
    migraphx::shape s1{migraphx::shape::int8_type, {1}};
    migraphx::shape s2{migraphx::shape::int8_type, {1280, 1000}};
    migraphx::shape s3{migraphx::shape::int8_type, {1000}};
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s5{migraphx::shape::int32_type, {1}};
    migraphx::shape s6{migraphx::shape::int32_type, {1280}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};
    migraphx::shape s8{migraphx::shape::float_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{0};

    auto create_module = [&]() {
        migraphx::module mm;
        auto db   = mm.add_parameter("db", s2);   // dot input b
        auto ab   = mm.add_parameter("ab", s3);   // add input b
        auto w    = mm.add_parameter("w", s4);    // conv weights (w)
        auto bias = mm.add_parameter("bias", s6); // bias
        auto x    = mm.add_parameter("x", s7);    // conv input (x)

        auto l1  = mm.add_literal(s1, zero);
        auto l2  = mm.add_literal(s1, zero);
        auto l3  = mm.add_literal(s1, zero);
        auto l4  = mm.add_literal(s1, zero);
        auto l5  = mm.add_literal(s1, zero);
        auto l6  = mm.add_literal(s5, zero);
        auto l7  = mm.add_literal(s1, zero);
        auto l8  = mm.add_literal(s1, zero);
        auto l9  = mm.add_literal(s8, scale);
        auto l10 = mm.add_literal(s8, scale);
        auto l11 = mm.add_literal(s8, scale);
        auto l12 = mm.add_literal(s8, scale);
        auto l13 = mm.add_literal(s8, scale);
        auto l14 = mm.add_literal(s8, scale);
        auto l15 = mm.add_literal(s8, scale);
        auto l16 = mm.add_literal(s8, scale);

        auto d1  = add_quantize_op(mm, "dequantizelinear", w, l16, l7);
        auto d2  = add_quantize_op(mm, "dequantizelinear", bias, l11, l6);
        auto d3  = add_quantize_op(mm, "dequantizelinear", ab, l9, l7);
        auto d4  = add_quantize_op(mm, "dequantizelinear", db, l11, l8);
        auto q1  = add_quantize_op(mm, "quantizelinear", x, l15, l5);
        auto d5  = add_quantize_op(mm, "dequantizelinear", q1, l15, l5);
        auto c1  = mm.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     d5,
                                     d1);
        auto bc1 = mm.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 1280, 7, 7}}}), d2);
        auto a1 = mm.add_instruction(migraphx::make_op("add"), c1, bc1);
        auto q2 = add_quantize_op(mm, "quantizelinear", a1, l14, l4);
        auto d6 = add_quantize_op(mm, "dequantizelinear", q2, l14, l4);
        auto ap = mm.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", "average"},
                                                        {"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"lengths", {7, 7}},
                                                        {"ceil_mode", 0}}),
                                     d6);
        auto q3 = add_quantize_op(mm, "quantizelinear", ap, l13, l3);
        auto d7 = add_quantize_op(mm, "dequantizelinear", q3, l13, l3);
        auto rs = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1}}}), d7);
        auto q4 = add_quantize_op(mm, "quantizelinear", rs, l12, l2);
        auto d8 = add_quantize_op(mm, "dequantizelinear", q4, l12, l2);
        auto dot =
            mm.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d8, d4);
        auto q5  = add_quantize_op(mm, "quantizelinear", dot, l10, l1);
        auto d9  = add_quantize_op(mm, "dequantizelinear", q5, l10, l1);
        auto mb1 = mm.add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1000}}}), d3);
        auto a2 = mm.add_instruction(migraphx::make_op("add"), d9, mb1);
        mm.add_return({a2});

        return mm;
    };

    auto mod1 = create_module();
    auto mod2 = create_module();

    run_pass(mod2);

    auto match_qdq = migraphx::match::name("dequantizelinear")(
        migraphx::match::arg(0)(migraphx::match::name("quantizelinear")));
    auto ins1 = migraphx::match::find_match(mod1, match_qdq);
    auto ins2 = migraphx::match::find_match(mod2, match_qdq);

    EXPECT((ins1.result != mod1.end()) and (ins2.result == mod2.end()));
    EXPECT(any_of(mod1, &is_convolution));
    EXPECT(none_of(mod2, &is_convolution));
    EXPECT(any_of(mod1, &is_dot));
    EXPECT(none_of(mod2, &is_dot));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
