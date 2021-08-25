#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>

#include "migraphx/shape.hpp"
#include "test.hpp"
#include <migraphx/half.hpp>

TEST_CASE(param_add)
{
    auto create_program_float = [](bool add_return = false) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = mm->add_parameter("x", s);
        auto p2  = mm->add_parameter("y", s);
        auto sum = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        if(add_return)
        {
            mm->add_return({sum});
        }

        return p;
    };

    auto create_program_half = [](bool add_return = false) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = mm->add_parameter("x", s);
        auto hp1 = mm->insert_instruction(std::next(p1), migraphx::make_op("convert"), p1);
        auto p2  = mm->add_parameter("y", s);
        auto hp2 = mm->insert_instruction(std::next(p2), migraphx::make_op("convert"), p2);
        auto hs  = mm->add_instruction(migraphx::make_op("add"), hp1, hp2);
        auto res = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            hs);
        if(add_return)
        {
            mm->add_return({res});
        }

        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1);
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1, {"add"});
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float(true);
        auto p2 = create_program_half(true);

        migraphx::quantize_fp16(p1);
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float(true);
        auto p2 = create_program_half(true);

        migraphx::quantize_fp16(p1, {"add"});
        EXPECT(p1 == p2);
    }
}

TEST_CASE(param_add_sub)
{
    auto create_program_float = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1   = mm->add_parameter("x", s);
        auto p2   = mm->add_parameter("y", s);
        auto sum  = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        auto diff = mm->add_instruction(migraphx::make_op("sub"), sum, p2);
        mm->add_instruction(migraphx::make_op("add"), diff, p1);

        return p;
    };

    auto create_program_half_add = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = mm->add_parameter("x", s);
        auto hp1 = mm->insert_instruction(
            std::next(p1),
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            p1);
        auto p2  = mm->add_parameter("y", s);
        auto hp2 = mm->insert_instruction(
            std::next(p2),
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            p2);
        auto hsum = mm->add_instruction(migraphx::make_op("add"), hp1, hp2);
        auto sum  = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            hsum);
        auto diff  = mm->add_instruction(migraphx::make_op("sub"), sum, p2);
        auto hdiff = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            diff);
        auto res = mm->add_instruction(migraphx::make_op("add"), hdiff, hp1);
        mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            res);

        return p;
    };

    auto create_program_half_sub = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1 = mm->add_parameter("x", s);
        mm->insert_instruction(
            std::next(p1),
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            p1);
        auto p2  = mm->add_parameter("y", s);
        auto hp2 = mm->insert_instruction(
            std::next(p2),
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            p2);
        auto sum  = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        auto hsum = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            sum);
        auto hdiff = mm->add_instruction(migraphx::make_op("sub"), hsum, hp2);
        auto diff  = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            hdiff);
        mm->add_instruction(migraphx::make_op("add"), diff, p1);

        return p;
    };

    auto create_program_half_all = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = mm->add_parameter("x", s);
        auto hp1 = mm->insert_instruction(
            std::next(p1),
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            p1);
        auto p2  = mm->add_parameter("y", s);
        auto hp2 = mm->insert_instruction(
            std::next(p2),
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            p2);
        auto hsum  = mm->add_instruction(migraphx::make_op("add"), hp1, hp2);
        auto hdiff = mm->add_instruction(migraphx::make_op("sub"), hsum, hp2);
        auto hres  = mm->add_instruction(migraphx::make_op("add"), hdiff, hp1);
        mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            hres);

        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half_add();

        migraphx::quantize_fp16(p1, {"add"});
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half_sub();

        migraphx::quantize_fp16(p1, {"sub"});
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half_all();

        migraphx::quantize_fp16(p1);
        migraphx::run_passes(*p1.get_main_module(), {migraphx::dead_code_elimination{}});

        EXPECT(p1 == p2);
    }
}

TEST_CASE(literal_add)
{
    auto create_program_float = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = mm->add_literal(migraphx::literal(s, data));
        auto l2 = mm->add_literal(migraphx::literal(s, data));
        mm->add_instruction(migraphx::make_op("add"), l1, l2);

        return p;
    };

    auto create_program_half = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::half_type, {2, 3}};
        std::vector<migraphx::half> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = mm->add_literal(migraphx::literal(s, data));
        auto l2 = mm->add_literal(migraphx::literal(s, data));
        auto hs = mm->add_instruction(migraphx::make_op("add"), l1, l2);
        mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            hs);

        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1, {"all"});
        migraphx::run_passes(*p1.get_main_module(),
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        migraphx::run_passes(*p2.get_main_module(),
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});

        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1, {"add"});
        migraphx::run_passes(*p1.get_main_module(),
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        migraphx::run_passes(*p2.get_main_module(),
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        EXPECT(p1 == p2);
    }
}

TEST_CASE(fp16_subgraph)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {1}};
        auto l1 = mm->add_literal(migraphx::literal(sd, {1}));
        auto l2 = mm->add_literal(migraphx::literal(sd, {2}));
        auto l3 = mm->add_literal(migraphx::literal(sd, {3}));
        migraphx::shape sx{migraphx::shape::float_type, {1, 4}};
        migraphx::shape sy{migraphx::shape::float_type, {3, 4}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto x    = mm->add_parameter("x", sx);
        auto y    = mm->add_parameter("y", sy);

        auto* then_mod = p.create_module("If_6_if");
        auto m1        = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), l1);
        auto add0 = then_mod->add_instruction(migraphx::make_op("add"), x, m1);
        auto m2   = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), l2);
        auto mul0  = then_mod->add_instruction(migraphx::make_op("mul"), y, m2);
        auto mfp16 = then_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), mul0);
        then_mod->add_return({add0, mul0, mfp16});

        auto* else_mod = p.create_module("If_6_else");
        auto me1       = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), l3);
        auto mul1 = else_mod->add_instruction(migraphx::make_op("mul"), x, me1);
        auto me2  = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), l3);
        auto add1  = else_mod->add_instruction(migraphx::make_op("add"), y, me2);
        auto afp16 = else_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), add1);
        else_mod->add_return({mul1, add1, afp16});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r0  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        auto r1  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), ret);
        auto r16 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), ret);
        mm->add_return({r0, r1, r16});

        return p;
    };

    auto create_fp16_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {1}};
        auto l1 = mm->add_literal(migraphx::literal(sd, {1}));
        auto l2 = mm->add_literal(migraphx::literal(sd, {2}));
        auto l3 = mm->add_literal(migraphx::literal(sd, {3}));
        migraphx::shape sx{migraphx::shape::float_type, {1, 4}};
        migraphx::shape sy{migraphx::shape::float_type, {3, 4}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto x    = mm->add_parameter("x", sx);
        auto y    = mm->add_parameter("y", sy);
        auto hl1  = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), l1);
        auto hl2 = mm->insert_instruction(
            std::next(l2),
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
            l2);
        auto hl3 = mm->insert_instruction(
            std::next(l3),
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
            l3);
        auto hx = mm->insert_instruction(
            std::next(x),
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
            x);
        auto hy = mm->insert_instruction(
            std::next(y),
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
            y);

        auto* then_mod = p.create_module("If_6_if");
        auto mhl1      = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), hl1);
        auto ad   = then_mod->add_instruction(migraphx::make_op("add"), hx, mhl1);
        auto mhl2 = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), hl2);
        auto mu      = then_mod->add_instruction(migraphx::make_op("mul"), hy, mhl2);
        auto mu_fp32 = then_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), mu);
        auto mu_fp16 = then_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), mu_fp32);
        then_mod->add_return({ad, mu, mu_fp16});

        auto* else_mod = p.create_module("If_6_else");
        auto mhl3      = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), hl3);
        auto mu1  = else_mod->add_instruction(migraphx::make_op("mul"), hx, mhl3);
        auto mhl4 = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), hl3);
        auto ad1     = else_mod->add_instruction(migraphx::make_op("add"), hy, mhl4);
        auto ad_fp32 = else_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), ad1);
        auto ad_fp16 = else_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), ad_fp32);
        else_mod->add_return({mu1, ad1, ad_fp16});

        auto iff = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto hr0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), iff);
        auto r0  = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), hr0);
        auto hr1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), iff);
        auto r1  = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), hr1);
        auto r2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), iff);
        r2      = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), r2);
        mm->add_return({r0, r1, r2});

        return p;
    };

    auto p1 = create_program();
    migraphx::quantize_fp16(p1);
    migraphx::run_passes(p1, {migraphx::dead_code_elimination{}});

    auto p2 = create_fp16_program();

    EXPECT(p1 == p2);
}

TEST_CASE(op_capture)
{
    auto test_func = [&](std::size_t ins_index, const std::vector<migraphx::argument>& args) {
        (void)ins_index;
        (void)args;
    };

    auto create_program_float = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s1{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};

        auto p1 = mm->add_parameter("x", s1);
        auto p2 = mm->add_parameter("y", s1);
        auto pb = mm->add_parameter("b", s2);
        auto pc = mm->add_parameter("c", s2);
        auto pa = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        auto ps = mm->add_instruction(migraphx::make_op("dot"), pa, pb, pc);
        mm->add_instruction(migraphx::make_op("dot"), pa, ps);

        return p;
    };

    auto create_program_op = [&] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s1{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};

        auto p1  = mm->add_parameter("x", s1);
        auto p2  = mm->add_parameter("y", s1);
        auto pb  = mm->add_parameter("b", s2);
        auto pc  = mm->add_parameter("c", s2);
        auto pa  = mm->add_instruction(migraphx::make_op("add"), p1, p2);
        auto opa = mm->add_instruction(migraphx::op::capture{0, test_func}, pa);
        auto opb = mm->add_instruction(migraphx::op::capture{1, test_func}, pb);
        auto opc = mm->add_instruction(migraphx::op::capture{2, test_func}, pc);
        auto ps  = mm->add_instruction(migraphx::make_op("dot"), opa, opb, opc);
        auto ops = mm->add_instruction(migraphx::op::capture{3, test_func}, ps);
        mm->add_instruction(migraphx::make_op("dot"), opa, ops);

        return p;
    };

    {
        auto p             = create_program_float();
        auto op_capture_p  = create_program_op();
        migraphx::target t = migraphx::ref::target{};
        migraphx::capture_arguments(p, t, {"dot", "convolution"});
        EXPECT(p == op_capture_p);
    }
}

TEST_CASE(op_capture_subgraph)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {2, 2, 4, 8}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 8, 6}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto a    = mm->add_parameter("a", sx);
        auto b    = mm->add_parameter("b", sy);

        migraphx::shape sd{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sw{migraphx::shape::float_type, {2, 2, 1, 1}};
        auto x = mm->add_parameter("x", sd);
        auto w = mm->add_parameter("w", sw);

        auto* then_mod = p.create_module("If_6_if");
        auto out1      = then_mod->add_instruction(migraphx::make_op("dot"), a, b);
        then_mod->add_return({out1});

        auto* else_mod = p.create_module("If_6_else");
        auto out2      = else_mod->add_instruction(migraphx::make_op("convolution"), x, w);
        else_mod->add_return({out2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({ret});

        return p;
    };

    auto create_program_op = [&] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {2, 2, 4, 8}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 8, 6}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto a    = mm->add_parameter("a", sx);
        auto b    = mm->add_parameter("b", sy);

        migraphx::shape sd{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sw{migraphx::shape::float_type, {2, 2, 1, 1}};
        auto x = mm->add_parameter("x", sd);
        auto w = mm->add_parameter("w", sw);

        auto* then_mod = p.create_module("If_6_if");
        auto ca   = then_mod->add_instruction(migraphx::make_op("capture", {{"ins_index", 0}}), a);
        auto cb   = then_mod->add_instruction(migraphx::make_op("capture", {{"ins_index", 1}}), b);
        auto out1 = then_mod->add_instruction(migraphx::make_op("dot"), ca, cb);
        then_mod->add_return({out1});

        auto* else_mod = p.create_module("If_6_else");
        auto cx   = else_mod->add_instruction(migraphx::make_op("capture", {{"ins_index", 2}}), x);
        auto cw   = else_mod->add_instruction(migraphx::make_op("capture", {{"ins_index", 3}}), w);
        auto out2 = else_mod->add_instruction(migraphx::make_op("convolution"), cx, cw);
        else_mod->add_return({out2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({ret});

        return p;
    };

    {
        auto p             = create_program();
        auto op_capture_p  = create_program_op();
        migraphx::target t = migraphx::ref::target{};
        migraphx::capture_arguments(p, t, {"dot", "convolution"});
        EXPECT(p == op_capture_p);
    }
}

TEST_CASE(dot_float)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);

        auto r = mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 2.0f}, {"beta", 1.5f}}), pa, pb, pc);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa      = mm->add_parameter("a", sa);
        auto pb      = mm->add_parameter("b", sb);
        auto pc      = mm->add_parameter("c", sc);
        auto zp_a    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_a = mm->add_literal(10.0f);
        scale_a      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_a);
        zp_a = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}),
                                   zp_a);
        auto quant_a = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);
        auto zp_b    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_b = mm->add_literal(10.0f);
        scale_b      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale_b);
        zp_b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}),
                                   zp_b);
        auto quant_b = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);
        auto quant   = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), quant_a, quant_b);
        std::vector<float> vec(sc.elements(), 200.0f);
        auto dc      = mm->add_literal(migraphx::literal(sc, vec));
        auto beta    = mm->add_literal(-0.0075f);
        auto mb_beta = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sc.lens()}}), beta);
        auto mc = mm->add_instruction(migraphx::make_op("mul"), mb_beta, pc);
        auto ic = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), mc);
        auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, dc, ic);
        mm->add_return({r});

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 0.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});

    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(dot_double_2args)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto r  = mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 2.0f}, {"beta", 1.5f}}), pa, pb);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);

        auto zp_a    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_a = mm->add_literal(10.0f);
        scale_a      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_a);
        zp_a = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}),
                                   zp_a);
        auto quant_a = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);
        auto zp_b    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_b = mm->add_literal(10.0f);
        scale_b      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale_b);
        zp_b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}),
                                   zp_b);
        auto quant_b = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);

        auto quant = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), quant_a, quant_b);
        std::vector<float> vec(sc.elements(), 200.0f);
        auto dc   = mm->add_literal(migraphx::literal(sc, vec));
        auto beta = mm->add_literal(int32_t(0));
        auto zp   = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sc.lens()}}), beta);
        auto fr = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, dc, zp);
        auto r  = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::double_type}}), fr);
        mm->add_return({r});
        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(dot_large_alpha_beta_float)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);
        auto r  = mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 20.0f}, {"beta", 50.5f}}), pa, pb, pc);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa      = mm->add_parameter("a", sa);
        auto pb      = mm->add_parameter("b", sb);
        auto pc      = mm->add_parameter("c", sc);
        auto zp_a    = mm->add_literal(static_cast<int8_t>(1));
        auto scale_a = mm->add_literal(10.0f);
        scale_a      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_a);
        zp_a = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}),
                                   zp_a);
        auto quant_a = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);
        auto zp_b    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_b = mm->add_literal(10.0f);
        scale_b      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale_b);
        zp_b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}),
                                   zp_b);
        auto quant_b = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);

        auto quant = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), quant_a, quant_b);

        std::vector<float> vec(sc.elements(), 2000.0f);
        auto dc      = mm->add_literal(migraphx::literal(sc, vec));
        auto beta    = mm->add_literal(-0.02525f);
        auto mb_beta = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sc.lens()}}), beta);
        auto mc = mm->add_instruction(migraphx::make_op("mul"), mb_beta, pc);
        auto ic = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), mc);
        auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, dc, ic);
        mm->add_return({r});

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(dot_large_alpha_beta_int32)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::int32_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::int32_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::int32_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);

        auto r = mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 20.0f}, {"beta", 50.0f}}), pa, pb, pc);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::int32_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::int32_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::int32_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);

        auto zp_a    = mm->add_literal(static_cast<int8_t>(1));
        auto scale_a = mm->add_literal(10.0f);
        scale_a      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_a);
        zp_a = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}),
                                   zp_a);
        auto quant_a = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);
        auto zp_b    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_b = mm->add_literal(10.0f);
        scale_b      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale_b);
        zp_b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}),
                                   zp_b);
        auto quant_b = mm->insert_instruction(
            std::next(zp_b), migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);

        auto quant = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), quant_a, quant_b);

        std::vector<float> vec(sc.elements(), 2000.0f);
        migraphx::shape s_scale{migraphx::shape::float_type, sc.lens()};
        auto dc      = mm->add_literal(migraphx::literal(s_scale, vec));
        auto beta    = mm->add_literal(-0.025f);
        auto mb_beta = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sc.lens()}}), beta);
        auto fc = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), pc);
        auto bc = mm->add_instruction(migraphx::make_op("mul"), mb_beta, fc);
        auto ic = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), bc);
        auto fdot = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, dc, ic);
        auto r    = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), fdot);
        mm->add_return({r});
        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(dot_int32_one_arg)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {16, 16}};
        auto pa = mm->add_parameter("a", s);
        auto r  = mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 20.0f}, {"beta", 50.0f}}), pa, pa);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {16, 16}};
        auto pa      = mm->add_parameter("a", s);
        auto zp_a    = mm->add_literal(static_cast<int8_t>(1));
        auto scale_a = mm->add_literal(1.0f);
        scale_a = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}),
                                      scale_a);
        zp_a    = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}),
                                   zp_a);
        auto quant_a = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);

        auto quant = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), quant_a, quant_a);

        std::vector<float> vec(s.elements(), 20.0f);
        migraphx::shape s_scale{migraphx::shape::float_type, s.lens()};
        auto dc = mm->add_literal(migraphx::literal(s_scale, vec));

        auto beta    = mm->add_literal(int32_t(0));
        auto mb_beta = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), beta);
        auto fdot = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, dc, mb_beta);
        auto r    = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), fdot);
        mm->add_return({r});

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{1.0f, 1.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(dot_int32)
{
    auto create_program = [](bool add_return = false) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::int32_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::int32_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::int32_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);

        auto res = mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 2.0f}, {"beta", 5.5f}}), pa, pb, pc);
        if(add_return)
        {
            mm->add_return({res});
        }

        return p;
    };

    auto create_int8_quantized_prog = [](bool add_return = false) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::int32_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::int32_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::int32_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);

        auto zp_a    = mm->add_literal(static_cast<int8_t>(1));
        auto scale_a = mm->add_literal(10.0f);
        scale_a      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}), scale_a);
        zp_a = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sa.lens()}}),
                                   zp_a);
        auto quant_a = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, scale_a, zp_a);
        auto zp_b    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_b = mm->add_literal(10.0f);
        scale_b      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale_b);
        zp_b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}),
                                   zp_b);
        auto quant_b = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);

        auto quant = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), quant_a, quant_b);

        std::vector<float> vec(sc.elements(), 200.0f);
        migraphx::shape s_scale{migraphx::shape::float_type, sc.lens()};
        auto dc      = mm->add_literal(migraphx::literal(s_scale, vec));
        auto beta    = mm->add_literal(-0.0275f);
        auto mb_beta = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sc.lens()}}), beta);
        auto fc = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), pc);
        auto bc = mm->add_instruction(migraphx::make_op("mul"), mb_beta, fc);
        auto ic = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), bc);
        auto fdot = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, dc, ic);
        auto r    = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), fdot);
        if(add_return)
        {
            mm->add_return({r});
        }
        else
        {
            mm->add_instruction(migraphx::make_op("identity"), r);
        }

        return p;
    };

    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    auto p_ret = create_program(true);
    migraphx::quantize_int8_impl(p_ret, quant_params, {"dot"});
    migraphx::run_passes(*p_ret.get_main_module(), {migraphx::dead_code_elimination{}});
    auto qp_ret = create_int8_quantized_prog(true);
    EXPECT(p_ret == qp_ret);

    auto p = create_program();
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
    auto qp = create_int8_quantized_prog();
    EXPECT(p == qp);
}

TEST_CASE(dot_float_convert)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::int8_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);

        auto fpa = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pa);
        auto r = mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 2.0f}, {"beta", 5.5f}}), fpa, pb);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::int8_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);

        auto zp_b    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_b = mm->add_literal(10.0f);
        scale_b      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}), scale_b);
        zp_b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sb.lens()}}),
                                   zp_b);
        auto quant_b = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, scale_b, zp_b);

        auto quant = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), pa, quant_b);

        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        std::vector<float> vec(sc.elements(), 20.0f);
        migraphx::shape s_scale{migraphx::shape::float_type, sc.lens()};
        auto dc      = mm->add_literal(migraphx::literal(s_scale, vec));
        auto beta    = mm->add_literal(int32_t(0));
        auto mb_beta = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sc.lens()}}), beta);
        auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, dc, mb_beta);
        mm->add_return({r});

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 1.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(conv_float)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto r = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        mm->add_return({r});

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {4, 3, 3, 3}};
        migraphx::shape sw{migraphx::shape::float_type, {4, 3, 3, 3}};
        auto px = mm->add_parameter("x", sx);
        auto pw = mm->add_parameter("w", sw);

        auto zp_x    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_x = mm->add_literal(10.0f);
        scale_x      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), scale_x);
        zp_x = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}),
                                   zp_x);
        auto quant_x = mm->add_instruction(migraphx::make_op("quantizelinear"), px, scale_x, zp_x);
        auto zp_w    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_w = mm->add_literal(10.0f);
        scale_w      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}), scale_w);
        zp_w = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}),
                                   zp_w);
        auto quant_w = mm->add_instruction(migraphx::make_op("quantizelinear"), pw, scale_w, zp_w);

        auto quant = mm->add_instruction(migraphx::make_op("quant_convolution"), quant_x, quant_w);

        migraphx::shape sc{migraphx::shape::float_type, {4, 4, 1, 1}};
        std::vector<float> vec(sc.elements(), 100.0f);
        migraphx::shape s_scale{migraphx::shape::float_type, sc.lens()};
        auto d_scale = mm->add_literal(migraphx::literal(s_scale, vec));
        auto r       = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, d_scale);
        mm->add_return({r});

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"convolution"});
    migraphx::run_passes(*p.get_main_module(), {migraphx::dead_code_elimination{}});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(conv_float_throw)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto r = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        mm->add_return({r});

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    test::throws([&] { migraphx::quantize_int8_impl(p, quant_params, {"add"}); });
}

TEST_CASE(conv_int32)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::int32_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::int32_type, {4, 3, 3, 3}});
        mm->add_instruction(migraphx::make_op("convolution"), input, weights);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;

        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::int32_type, {4, 3, 3, 3}};
        migraphx::shape sw{migraphx::shape::int32_type, {4, 3, 3, 3}};
        auto px = mm->add_parameter("x", sx);
        auto pw = mm->add_parameter("w", sw);

        auto zp_x    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_x = mm->add_literal(10.0f);
        scale_x      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), scale_x);
        zp_x = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}),
                                   zp_x);
        auto quant_x = mm->add_instruction(migraphx::make_op("quantizelinear"), px, scale_x, zp_x);
        auto zp_w    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_w = mm->add_literal(10.0f);
        scale_w      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}), scale_w);
        zp_w = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}),
                                   zp_w);
        auto quant_w = mm->add_instruction(migraphx::make_op("quantizelinear"), pw, scale_w, zp_w);

        auto quant = mm->add_instruction(migraphx::make_op("quant_convolution"), quant_x, quant_w);
        migraphx::shape sc{migraphx::shape::float_type, {4, 4, 1, 1}};
        std::vector<float> vec(sc.elements(), 100.0f);
        migraphx::shape s_scale{migraphx::shape::float_type, sc.lens()};
        auto d_scale = mm->add_literal(migraphx::literal(s_scale, vec));
        auto fr      = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, d_scale);
        auto r       = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), fr);
        mm->add_instruction(migraphx::make_op("identity"), r);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"convolution"});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(conv_half)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        mm->add_instruction(migraphx::make_op("convolution"), input, weights);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::half_type, {4, 3, 3, 3}};
        migraphx::shape sw{migraphx::shape::half_type, {4, 3, 3, 3}};
        auto px = mm->add_parameter("x", sx);
        auto pw = mm->add_parameter("w", sw);

        auto zp_x    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_x = mm->add_literal(10.0f);
        scale_x      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), scale_x);
        zp_x = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}),
                                   zp_x);
        auto quant_x = mm->add_instruction(migraphx::make_op("quantizelinear"), px, scale_x, zp_x);
        auto zp_w    = mm->add_literal(static_cast<int8_t>(0));
        auto scale_w = mm->add_literal(10.0f);
        scale_w      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}), scale_w);
        zp_w = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}),
                                   zp_w);
        auto quant_w = mm->add_instruction(migraphx::make_op("quantizelinear"), pw, scale_w, zp_w);

        auto quant = mm->add_instruction(migraphx::make_op("quant_convolution"), quant_x, quant_w);
        migraphx::shape sc{migraphx::shape::float_type, {4, 4, 1, 1}};
        std::vector<float> vec(sc.elements(), 100.0f);
        migraphx::shape s_scale{migraphx::shape::float_type, sc.lens()};
        auto d_scale = mm->add_literal(migraphx::literal(s_scale, vec));
        auto fr      = mm->add_instruction(migraphx::make_op("dequantizelinear"), quant, d_scale);
        auto r       = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), fr);
        mm->add_instruction(migraphx::make_op("identity"), r);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"convolution"});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

TEST_CASE(target_copy)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::parameter_map& m_in,
                       std::vector<float>& res) {
        p.compile(t);
        migraphx::parameter_map m;
        for(auto&& x : p.get_parameter_shapes())
        {
            if(m_in.count(x.first) > 0)
            {
                m[x.first] = t.copy_to(m_in[x.first]);
            }
            else
            {
                m[x.first] = t.allocate(x.second);
            }
        }

        auto result = t.copy_from(p.eval(m).back());
        result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
    };

    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto p1 = mm->add_parameter("x", s);
        auto p2 = mm->add_parameter("y", s);
        mm->add_instruction(migraphx::make_op("add"), p1, p2);

        return p;
    };

    {
        auto p = create_program();
        migraphx::parameter_map m;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        m["x"] = migraphx::generate_argument(s);
        std::vector<float> ref_result;
        migraphx::target ref_t = migraphx::ref::target{};
        run_prog(p, ref_t, m, ref_result);

        std::vector<float> orig_result;
        run_prog(p, ref_t, m, orig_result);

        EXPECT(migraphx::verify_range(ref_result, orig_result));
    }
}

TEST_CASE(int8_quantization_dot)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::parameter_map& m_in,
                       std::vector<float>& res,
                       bool b_quantize = false) {
        if(b_quantize)
        {
            std::vector<migraphx::parameter_map> cali_data;
            cali_data.push_back(m_in);
            migraphx::quantize_int8(p, t, cali_data);
        }
        p.compile(t);
        migraphx::parameter_map m;
        for(auto&& x : p.get_parameter_shapes())
        {
            if(m_in.count(x.first) > 0)
            {
                m[x.first] = t.copy_to(m_in[x.first]);
            }
            else
            {
                m[x.first] = t.allocate(x.second);
            }
        }

        auto result = t.copy_from(p.eval(m).back());
        result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
    };

    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);
        auto r  = mm->add_instruction(migraphx::make_op("dot"), pa, pb, pc);
        mm->add_return({r});

        return p;
    };

    {
        auto p = create_program();
        migraphx::parameter_map m;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        m["a"] = migraphx::generate_argument(sa, get_hash(std::string("a")));
        m["b"] = migraphx::generate_argument(sb, get_hash(std::string("b")));
        m["c"] = migraphx::generate_argument(sc, get_hash(std::string("c")));
        std::vector<float> quant_result;
        migraphx::target ref_t = migraphx::ref::target{};
        run_prog(p, ref_t, m, quant_result, true);

        std::vector<float> no_quant_result;
        run_prog(p, ref_t, m, no_quant_result);

        EXPECT(migraphx::verify_range(quant_result, no_quant_result, 200000));
    }
}

TEST_CASE(int8_quantization_conv)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       std::vector<float>& res,
                       bool b_quantize = false) {
        if(b_quantize)
        {
            std::vector<migraphx::parameter_map> cali_data;
            migraphx::quantize_int8(p, t, cali_data);
        }
        p.compile(t);
        migraphx::parameter_map m;

        auto result = t.copy_from(p.eval(m).back());
        result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
    };

    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {4, 2, 2, 2}};
        migraphx::shape sw{migraphx::shape::float_type, {4, 2, 2, 2}};
        std::vector<float> v(sx.elements(), 0.5f);
        auto input   = mm->add_literal(migraphx::literal(sx, v));
        auto weights = mm->add_literal(migraphx::literal(sw, v));
        mm->add_instruction(migraphx::make_op("convolution"), input, weights);

        return p;
    };

    {
        auto p = create_program();
        std::vector<float> quant_result;
        migraphx::target ref_t = migraphx::ref::target{};
        run_prog(p, ref_t, quant_result, true);

        std::vector<float> no_quant_result;
        run_prog(p, ref_t, no_quant_result);

        EXPECT(migraphx::verify_range(quant_result, no_quant_result));
    }
}

TEST_CASE(int8_subgraph)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {2, 2, 4, 8}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 8, 6}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto a    = mm->add_parameter("a", sx);
        auto b    = mm->add_parameter("b", sy);

        migraphx::shape sd{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sw{migraphx::shape::float_type, {2, 2, 1, 1}};
        auto x = mm->add_parameter("x", sd);
        auto w = mm->add_parameter("w", sw);

        auto* then_mod = p.create_module("If_6_if");
        auto out1      = then_mod->add_instruction(migraphx::make_op("dot"), a, b);
        then_mod->add_return({out1});

        auto* else_mod = p.create_module("If_6_else");
        auto out2      = else_mod->add_instruction(migraphx::make_op("convolution"), x, w);
        else_mod->add_return({out2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({ret});

        return p;
    };

    auto create_int8_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{migraphx::shape::float_type, {2, 2, 4, 8}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 8, 6}};
        migraphx::shape sout{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto a    = mm->add_parameter("a", sx);
        auto b    = mm->add_parameter("b", sy);

        // then submod
        auto* then_mod = p.create_module("If_6_if");
        auto zp1       = then_mod->add_literal(static_cast<int8_t>(0));
        auto s1        = then_mod->add_literal(10.0f);
        s1             = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), s1);
        zp1 = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sx.lens()}}), zp1);
        auto qa  = then_mod->add_instruction(migraphx::make_op("quantizelinear"), a, s1, zp1);
        auto zp2 = then_mod->add_literal(static_cast<int8_t>(0));
        auto s2  = then_mod->add_literal(10.0f);
        s2       = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sy.lens()}}), s2);
        zp2 = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sy.lens()}}), zp2);
        auto qb = then_mod->add_instruction(migraphx::make_op("quantizelinear"), b, s2, zp2);
        auto qdot =
            then_mod->add_instruction(migraphx::make_op("quant_dot", {{"beta", 0}}), qa, qb);
        std::vector<float> vec(sout.elements(), 100.0f);
        auto so  = then_mod->add_literal(migraphx::literal(sout, vec));
        auto zpo = then_mod->add_literal(int32_t(0));
        zpo      = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sout.lens()}}), zpo);
        auto r = then_mod->add_instruction(migraphx::make_op("dequantizelinear"), qdot, so, zpo);
        then_mod->add_return({r});

        migraphx::shape sd{migraphx::shape::float_type, {2, 2, 4, 6}};
        migraphx::shape sw{migraphx::shape::float_type, {2, 2, 1, 1}};
        auto x = mm->add_parameter("x", sd);
        auto w = mm->add_parameter("w", sw);
        // else submod
        auto* else_mod = p.create_module("If_6_else");
        auto zpx       = else_mod->add_literal(static_cast<int8_t>(0));
        auto sax       = else_mod->add_literal(2.0f);
        sax            = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sd.lens()}}), sax);
        zpx = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sd.lens()}}), zpx);
        auto qx  = else_mod->add_instruction(migraphx::make_op("quantizelinear"), x, sax, zpx);
        auto zpw = else_mod->add_literal(static_cast<int8_t>(0));
        auto ssw = else_mod->add_literal(1.66667f);
        ssw      = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}), ssw);
        zpw = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sw.lens()}}), zpw);
        auto qw    = else_mod->add_instruction(migraphx::make_op("quantizelinear"), w, ssw, zpw);
        auto qconv = else_mod->add_instruction(migraphx::make_op("quant_convolution"), qx, qw);
        auto so1   = else_mod->add_literal(migraphx::literal(sout, vec));
        auto r1    = else_mod->add_instruction(migraphx::make_op("dequantizelinear"), qconv, so1);
        else_mod->add_return({r1});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({ret});

        return p;
    };

    auto p1 = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 0.0f}, {0.1f, 0.0f}, {0.5f, 0.0f}, {0.6f, 0.0f}};
    migraphx::quantize_int8_impl(p1, quant_params, {"convolution", "dot"});
    migraphx::run_passes(p1, {migraphx::dead_code_elimination{}});

    auto p2 = create_int8_program();
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
