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
    migraphx::shape ss{migraphx::shape::float_type, {1}};
    migraphx::shape zs{migraphx::shape::int8_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{0};

    migraphx::module m1;
    {
        auto t1 = m1.add_parameter("t1", sh1);
        auto t2 = m1.add_parameter("t2", sh2);
        auto sc = m1.add_literal(ss, scale);
        auto z  = m1.add_literal(zs, zero);

        auto q1 = add_quantize_op(m1, "quantizelinear", t1, sc, z);
        auto d1 = add_quantize_op(m1, "dequantizelinear", q1, sc, z);
        auto q2 = add_quantize_op(m1, "quantizelinear", t2, sc, z);
        auto d2 = add_quantize_op(m1, "dequantizelinear", q2, sc, z);
        auto dot =
            m1.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d1, d2);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1  = m2.add_parameter("t1", sh1);
        auto t2  = m2.add_parameter("t2", sh2);
        auto sc  = m2.add_literal(ss, scale);
        auto z   = m2.add_literal(zs, zero);
        auto sc1 = m2.add_literal(static_cast<float>(4));
        auto z1  = m2.add_literal(0);

        auto q1 = add_quantize_op(m2, "quantizelinear", t1, sc, z);
        auto q2 = add_quantize_op(m2, "quantizelinear", t2, sc, z);
        auto dot =
            m2.add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), q1, q2);
        auto d3 = add_quantize_op(m2, "dequantizelinear", dot, sc1, z1);
        m2.add_return({d3});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_non_zero_point)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};
    migraphx::shape ss{migraphx::shape::float_type, {1}};
    migraphx::shape zs{migraphx::shape::int8_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{1};

    migraphx::module m1;
    {
        auto t1 = m1.add_parameter("t1", sh1);
        auto t2 = m1.add_parameter("t2", sh2);
        auto sc = m1.add_literal(ss, scale);
        auto z  = m1.add_literal(zs, zero);

        auto q1 = add_quantize_op(m1, "quantizelinear", t1, sc, z);
        auto d1 = add_quantize_op(m1, "dequantizelinear", q1, sc, z);
        auto q2 = add_quantize_op(m1, "quantizelinear", t2, sc, z);
        auto d2 = add_quantize_op(m1, "dequantizelinear", q2, sc, z);
        auto dot =
            m1.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d1, d2);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1 = m2.add_parameter("t1", sh1);
        auto t2 = m2.add_parameter("t2", sh2);

        auto dot =
            m2.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), t1, t2);
        m2.add_return({dot});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_uint8)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};
    migraphx::shape ss{migraphx::shape::float_type, {1}};
    migraphx::shape zs{migraphx::shape::uint8_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{1};

    migraphx::module m1;
    {
        auto t1 = m1.add_parameter("t1", sh1);
        auto t2 = m1.add_parameter("t2", sh2);
        auto sc = m1.add_literal(ss, scale);
        auto z  = m1.add_literal(zs, zero);

        auto q1 = add_quantize_op(m1, "quantizelinear", t1, sc, z);
        auto d1 = add_quantize_op(m1, "dequantizelinear", q1, sc, z);
        auto q2 = add_quantize_op(m1, "quantizelinear", t2, sc, z);
        auto d2 = add_quantize_op(m1, "dequantizelinear", q2, sc, z);
        auto dot =
            m1.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d1, d2);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1 = m2.add_parameter("t1", sh1);
        auto t2 = m2.add_parameter("t2", sh2);

        auto dot =
            m2.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), t1, t2);
        m2.add_return({dot});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_add)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};
    migraphx::shape sh3{migraphx::shape::float_type, {1280, 1024}};
    migraphx::shape ss{migraphx::shape::float_type, {1}};
    migraphx::shape zs{migraphx::shape::int8_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{0};

    migraphx::module m1;
    {
        auto t1 = m1.add_parameter("t1", sh1);
        auto t2 = m1.add_parameter("t2", sh2);
        auto ab = m1.add_parameter("ab", sh3);
        auto sc = m1.add_literal(ss, scale);
        auto z  = m1.add_literal(zs, zero);

        auto q1 = add_quantize_op(m1, "quantizelinear", t1, sc, z);
        auto d1 = add_quantize_op(m1, "dequantizelinear", q1, sc, z);
        auto q2 = add_quantize_op(m1, "quantizelinear", t2, sc, z);
        auto d2 = add_quantize_op(m1, "dequantizelinear", q2, sc, z);
        auto dot =
            m1.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d1, d2);
        auto q3  = add_quantize_op(m1, "quantizelinear", dot, sc, z);
        auto d3  = add_quantize_op(m1, "dequantizelinear", q3, sc, z);
        auto add = m1.add_instruction(migraphx::make_op("add"), d3, ab);
        m1.add_return({add});
    }

    migraphx::module m2;
    {
        auto t1  = m2.add_parameter("t1", sh1);
        auto t2  = m2.add_parameter("t2", sh2);
        auto ab  = m2.add_parameter("ab", sh3);
        auto sc  = m2.add_literal(ss, scale);
        auto z   = m2.add_literal(zs, zero);
        auto sc1 = m2.add_literal(static_cast<float>(4));
        auto z1  = m2.add_literal(0);

        auto q1 = add_quantize_op(m2, "quantizelinear", t1, sc, z);
        auto q2 = add_quantize_op(m2, "quantizelinear", t2, sc, z);
        auto dot =
            m2.add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), q1, q2);
        auto d3  = add_quantize_op(m2, "dequantizelinear", dot, sc1, z1);
        auto add = m2.add_instruction(migraphx::make_op("add"), d3, ab);
        m2.add_return({add});
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
        auto input   = m1.add_parameter("input", s7);
        auto weights = m1.add_parameter("weights", s4);
        auto l1      = m1.add_literal(s1, zero);
        auto l2      = m1.add_literal(s8, scale);

        auto d1 = add_quantize_op(m1, "dequantizelinear", weights, l2, l1);
        auto q1 = add_quantize_op(m1, "quantizelinear", input, l2, l1);
        auto d5 = add_quantize_op(m1, "dequantizelinear", q1, l2, l1);
        auto c1 = m1.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     d5,
                                     d1);
        m1.add_return({c1});
    }

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s7);
        auto weights = m2.add_parameter("weights", s4);
        auto l1      = m2.add_literal(s1, zero);
        auto l2      = m2.add_literal(s8, scale);
        auto sc1     = m2.add_literal(static_cast<float>(0.25));
        auto z1      = m2.add_literal(0);

        auto q1 = add_quantize_op(m2, "quantizelinear", input, l2, l1);
        auto c1 = m2.add_instruction(migraphx::make_op("quant_convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     q1,
                                     weights);
        auto d6 = add_quantize_op(m2, "dequantizelinear", c1, sc1, z1);
        m2.add_return({d6});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(conv_bias_add)
{
    migraphx::shape s1{migraphx::shape::int8_type, {1}};
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s6{migraphx::shape::int32_type, {1280}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};
    migraphx::shape s8{migraphx::shape::float_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{0};

    migraphx::module m1;
    {
        auto input   = m1.add_parameter("input", s7);
        auto weights = m1.add_parameter("weights", s4);
        auto bias    = m1.add_parameter("bias", s6);
        auto l1      = m1.add_literal(s1, zero);
        auto l2      = m1.add_literal(s8, scale);

        auto d1 = add_quantize_op(m1, "dequantizelinear", weights, l2, l1);
        auto d2 = add_quantize_op(m1, "dequantizelinear", bias, l2, l1);
        auto q1 = add_quantize_op(m1, "quantizelinear", input, l2, l1);
        auto d5 = add_quantize_op(m1, "dequantizelinear", q1, l2, l1);
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
        m1.add_return({a1});
    }

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s7);
        auto weights = m2.add_parameter("weights", s4);
        auto bias    = m2.add_parameter("bias", s6);
        auto l1      = m2.add_literal(s1, zero);
        auto l2      = m2.add_literal(s8, scale);
        auto sc1     = m2.add_literal(static_cast<float>(0.25));
        auto z1      = m2.add_literal(0);

        auto d2 = add_quantize_op(m2, "dequantizelinear", bias, l2, l1);
        auto q1 = add_quantize_op(m2, "quantizelinear", input, l2, l1);
        auto c1 = m2.add_instruction(migraphx::make_op("quant_convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     q1,
                                     weights);
        auto b1 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 1280, 7, 7}}}), d2);
        auto d6 = add_quantize_op(m2, "dequantizelinear", c1, sc1, z1);
        auto a1 = m2.add_instruction(migraphx::make_op("add"), d6, b1);
        m2.add_return({a1});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(conv_pooling_dot)
{
    migraphx::shape s1{migraphx::shape::int8_type, {1}};
    migraphx::shape s2{migraphx::shape::int8_type, {1280, 1000}};
    migraphx::shape s3{migraphx::shape::int8_type, {1000}};
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s6{migraphx::shape::int32_type, {1280}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};
    migraphx::shape s8{migraphx::shape::float_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{0};

    migraphx::module m1;
    {
        auto db      = m1.add_parameter("db", s2); // dot input b
        auto ab      = m1.add_parameter("ab", s3); // add input b
        auto weights = m1.add_parameter("weights", s4);
        auto bias    = m1.add_parameter("bias", s6);
        auto input   = m1.add_parameter("input", s7);

        auto l1 = m1.add_literal(s1, zero);
        auto l2 = m1.add_literal(s8, scale);

        auto d1  = add_quantize_op(m1, "dequantizelinear", weights, l2, l1);
        auto d2  = add_quantize_op(m1, "dequantizelinear", bias, l2, l1);
        auto d3  = add_quantize_op(m1, "dequantizelinear", ab, l2, l1);
        auto d4  = add_quantize_op(m1, "dequantizelinear", db, l2, l1);
        auto q1  = add_quantize_op(m1, "quantizelinear", input, l2, l1);
        auto d5  = add_quantize_op(m1, "dequantizelinear", q1, l2, l1);
        auto c1  = m1.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     d5,
                                     d1);
        auto bc1 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 1280, 7, 7}}}), d2);
        auto a1 = m1.add_instruction(migraphx::make_op("add"), c1, bc1);
        auto ap = m1.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", "average"},
                                                        {"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"lengths", {7, 7}},
                                                        {"ceil_mode", 0}}),
                                     a1);
        auto fl = m1.add_instruction(migraphx::make_op("flatten", {{"axis", 1}}), ap);
        auto q4 = add_quantize_op(m1, "quantizelinear", fl, l2, l1);
        auto d8 = add_quantize_op(m1, "dequantizelinear", q4, l2, l1);
        auto dot =
            m1.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d8, d4);
        auto q5  = add_quantize_op(m1, "quantizelinear", dot, l2, l1);
        auto d9  = add_quantize_op(m1, "dequantizelinear", q5, l2, l1);
        auto mb1 = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1000}}}), d3);
        auto a2 = m1.add_instruction(migraphx::make_op("add"), d9, mb1);
        m1.add_return({a2});
    }

    migraphx::module m2;
    {
        auto db      = m2.add_parameter("db", s2); // dot input b
        auto ab      = m2.add_parameter("ab", s3); // add input b
        auto weights = m2.add_parameter("weights", s4);
        auto bias    = m2.add_parameter("bias", s6);
        auto input   = m2.add_parameter("input", s7);

        auto l1  = m2.add_literal(s1, zero);
        auto l2  = m2.add_literal(s8, scale);
        auto sc1 = m2.add_literal(static_cast<float>(0.25));
        auto z1  = m2.add_literal(0);
        auto sc2 = m2.add_literal(static_cast<float>(4));
        auto z2  = m2.add_literal(0);

        auto d2  = add_quantize_op(m2, "dequantizelinear", bias, l2, l1);
        auto d3  = add_quantize_op(m2, "dequantizelinear", ab, l2, l1);
        auto q1  = add_quantize_op(m2, "quantizelinear", input, l2, l1);
        auto c1  = m2.add_instruction(migraphx::make_op("quant_convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     q1,
                                     weights);
        auto bc1 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"dims", {1, 1280, 7, 7}}}), d2);
        auto d5 = add_quantize_op(m2, "dequantizelinear", c1, sc1, z1);
        auto a1 = m2.add_instruction(migraphx::make_op("add"), d5, bc1);
        auto ap = m2.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", "average"},
                                                        {"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"lengths", {7, 7}},
                                                        {"ceil_mode", 0}}),
                                     a1);
        auto fl = m2.add_instruction(migraphx::make_op("flatten", {{"axis", 1}}), ap);
        auto q4 = add_quantize_op(m2, "quantizelinear", fl, l2, l1);
        auto dot =
            m2.add_instruction(migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), q4, db);
        auto d9  = add_quantize_op(m2, "dequantizelinear", dot, sc2, z2);
        auto mb1 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"output_lens", {1, 1000}}}), d3);
        auto a2 = m2.add_instruction(migraphx::make_op("add"), d9, mb1);
        m2.add_return({a2});
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
    migraphx::shape s6{migraphx::shape::int32_type, {1280}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};
    migraphx::shape s8{migraphx::shape::float_type, {1}};

    std::vector<float> scale{0.5};
    std::vector<int> zero{0};

    auto create_module = [&]() {
        migraphx::module mm;
        auto db      = mm.add_parameter("db", s2); // dot input b
        auto ab      = mm.add_parameter("ab", s3); // add input b
        auto weights = mm.add_parameter("weights", s4);
        auto bias    = mm.add_parameter("bias", s6);
        auto input   = mm.add_parameter("input", s7);

        auto l1 = mm.add_literal(s1, zero);
        auto l2 = mm.add_literal(s8, scale);

        auto d1  = add_quantize_op(mm, "dequantizelinear", weights, l2, l1);
        auto d2  = add_quantize_op(mm, "dequantizelinear", bias, l2, l1);
        auto d3  = add_quantize_op(mm, "dequantizelinear", ab, l2, l1);
        auto d4  = add_quantize_op(mm, "dequantizelinear", db, l2, l1);
        auto q1  = add_quantize_op(mm, "quantizelinear", input, l2, l1);
        auto d5  = add_quantize_op(mm, "dequantizelinear", q1, l2, l1);
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
        auto q2 = add_quantize_op(mm, "quantizelinear", a1, l2, l1);
        auto d6 = add_quantize_op(mm, "dequantizelinear", q2, l2, l1);
        auto ap = mm.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", "average"},
                                                        {"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"lengths", {7, 7}},
                                                        {"ceil_mode", 0}}),
                                     d6);
        auto q3 = add_quantize_op(mm, "quantizelinear", ap, l2, l1);
        auto d7 = add_quantize_op(mm, "dequantizelinear", q3, l2, l1);
        auto rs = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1}}}), d7);
        auto q4 = add_quantize_op(mm, "quantizelinear", rs, l2, l1);
        auto d8 = add_quantize_op(mm, "dequantizelinear", q4, l2, l1);
        auto dot =
            mm.add_instruction(migraphx::make_op("dot", {{"alpha", 1}, {"beta", 0}}), d8, d4);
        auto q5  = add_quantize_op(mm, "quantizelinear", dot, l2, l1);
        auto d9  = add_quantize_op(mm, "dequantizelinear", q5, l2, l1);
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
