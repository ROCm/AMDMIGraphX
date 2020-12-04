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

#include "test.hpp"
#include <migraphx/half.hpp>

migraphx::instruction_ref
create_clip_op(migraphx::program& p, float max, float min, migraphx::instruction_ref input)
{
    auto* mm        = p.get_main_module();
    auto input_lens = input->get_shape().lens();
    auto max_val    = mm->add_literal(max);
    auto min_val    = mm->add_literal(min);
    max_val         = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", input_lens}}), max_val);
    min_val = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"output_lens", input_lens}}), min_val);
    return mm->add_instruction(migraphx::make_op("clip"), input, min_val, max_val);
}

migraphx::instruction_ref create_clip_op(migraphx::instruction_ref insert_loc,
                                         migraphx::program& p,
                                         float max,
                                         float min,
                                         migraphx::instruction_ref input)
{
    auto* mm        = p.get_main_module();
    auto input_lens = input->get_shape().lens();
    auto max_val    = mm->add_literal(max);
    auto min_val    = mm->add_literal(min);
    max_val         = mm->insert_instruction(
        insert_loc, migraphx::make_op("multibroadcast", {{"output_lens", input_lens}}), max_val);
    min_val = mm->insert_instruction(
        insert_loc, migraphx::make_op("multibroadcast", {{"output_lens", input_lens}}), min_val);
    return mm->insert_instruction(insert_loc, migraphx::make_op("clip"), input, min_val, max_val);
}

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
        auto p1  = mm->add_parameter("x", s);
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
        auto opb = mm->insert_instruction(std::next(pb), migraphx::op::capture{1, test_func}, pb);
        auto opc = mm->insert_instruction(std::next(pc), migraphx::op::capture{2, test_func}, pc);
        auto opa = mm->add_instruction(migraphx::op::capture{0, test_func}, pa);
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

        mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 2.0f}, {"beta", 1.5f}}), pa, pb, pc);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fa = mm->add_literal(migraphx::literal(sa, vfa));
        auto ma = mm->add_instruction(migraphx::make_op("mul"), fa, pa);
        auto ra = mm->add_instruction(migraphx::make_op("round"), ma);
        auto ca = create_clip_op(p, 127.0f, -128.0f, ra);
        auto qa = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = mm->add_literal(migraphx::literal(sb, vfb));
        auto mb = mm->insert_instruction(insert_loc, migraphx::make_op("mul"), fb, pb);
        auto rb = mm->insert_instruction(insert_loc, migraphx::make_op("round"), mb);
        auto cb = create_clip_op(insert_loc, p, 127.0f, -128.0f, rb);
        auto qb = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cb);

        auto qdot = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), qa, qb);
        auto fdot = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            qdot);
        std::vector<float> v_alpha(fdot->get_shape().elements(), 200.0f);
        auto new_alpha = mm->add_literal(migraphx::literal(fdot->get_shape(), v_alpha));
        auto alpha_ab  = mm->add_instruction(migraphx::make_op("mul"), new_alpha, fdot);
        std::vector<float> v_beta(pc->get_shape().elements(), 1.5f);
        auto beta   = mm->add_literal(migraphx::literal(pc->get_shape(), v_beta));
        auto beta_c = mm->add_instruction(migraphx::make_op("mul"), beta, pc);
        mm->add_instruction(migraphx::make_op("add"), alpha_ab, beta_c);

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

        mm->add_instruction(migraphx::make_op("dot", {{"alpha", 2.0f}, {"beta", 1.5f}}), pa, pb);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::double_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fpa = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pa);
        auto fa = mm->add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vfa));
        auto ma = mm->add_instruction(migraphx::make_op("mul"), fa, fpa);
        auto ra = mm->add_instruction(migraphx::make_op("round"), ma);
        auto ca = create_clip_op(p, 127.0f, -128.0f, ra);
        auto qa = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        auto fpb        = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = mm->add_literal(migraphx::literal({migraphx::shape::float_type, sb.lens()}, vfb));
        auto mb = mm->insert_instruction(insert_loc, migraphx::make_op("mul"), fb, fpb);
        auto rb = mm->insert_instruction(insert_loc, migraphx::make_op("round"), mb);
        auto cb = create_clip_op(insert_loc, p, 127.0f, -128.0f, rb);
        auto qb = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cb);

        auto qdot = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), qa, qb);
        auto fdot = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            qdot);
        std::vector<float> v_alpha(fdot->get_shape().elements(), 200.0f);
        auto new_alpha = mm->add_literal(migraphx::literal(fdot->get_shape(), v_alpha));
        auto alpha_ab  = mm->add_instruction(migraphx::make_op("mul"), new_alpha, fdot);
        mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
            alpha_ab);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
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

        mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 20.0f}, {"beta", 50.5f}}), pa, pb, pc);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        auto pc = mm->add_parameter("c", sc);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fa = mm->add_literal(migraphx::literal(sa, vfa));
        auto ma = mm->add_instruction(migraphx::make_op("mul"), fa, pa);
        // add the shift
        std::vector<float> vsa(sa.elements(), 1.0f);
        auto sfta = mm->add_literal(migraphx::literal(sa, vsa));
        auto msa  = mm->add_instruction(migraphx::make_op("add"), sfta, ma);
        auto ra   = mm->add_instruction(migraphx::make_op("round"), msa);
        auto ca   = create_clip_op(p, 127.0f, -128.0f, ra);
        auto qa   = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = mm->add_literal(migraphx::literal(sb, vfb));
        auto mb = mm->insert_instruction(insert_loc, migraphx::make_op("mul"), fb, pb);
        auto rb = mm->insert_instruction(insert_loc, migraphx::make_op("round"), mb);
        auto cb = create_clip_op(insert_loc, p, 127.0f, -128.0f, rb);
        auto qb = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cb);

        // quantize parameter c to int32 type
        auto qc = mm->insert_instruction(
            std::next(pc),
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int32_type)}}),
            pc);

        auto qdot = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 2000}, {"beta", 51}}), qa, qb, qc);
        mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            qdot);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
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

        mm->add_instruction(
            migraphx::make_op("dot", {{"alpha", 20.0f}, {"beta", 50.0f}}), pa, pb, pc);

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
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fa = mm->add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vfa));
        auto conv_a = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pa);
        auto ma = mm->add_instruction(migraphx::make_op("mul"), fa, conv_a);

        // add the shift
        std::vector<float> vsa(sa.elements(), 1.0f);
        auto sfta =
            mm->add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vsa));
        auto msa = mm->add_instruction(migraphx::make_op("add"), sfta, ma);
        auto ra  = mm->add_instruction(migraphx::make_op("round"), msa);
        auto ca  = create_clip_op(p, 127.0f, -128.0f, ra);
        auto qa  = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = mm->add_literal(migraphx::literal({migraphx::shape::float_type, sb.lens()}, vfb));
        auto conv_b = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pb);
        auto mb = mm->insert_instruction(insert_loc, migraphx::make_op("mul"), fb, conv_b);
        auto rb = mm->insert_instruction(insert_loc, migraphx::make_op("round"), mb);
        auto cb = create_clip_op(insert_loc, p, 127.0f, -128.0f, rb);
        auto qb = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cb);

        mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 2000}, {"beta", 50}}), qa, qb, pc);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
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

        mm->add_instruction(migraphx::make_op("dot", {{"alpha", 20.0f}, {"beta", 50.0f}}), pa, pa);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {16, 16}};
        auto pa = mm->add_parameter("a", s);

        // add the shift
        auto fpa = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pa);
        std::vector<float> vsa(s.elements(), 1.0f);
        auto sfta =
            mm->add_literal(migraphx::literal({migraphx::shape::float_type, s.lens()}, vsa));
        auto msa = mm->add_instruction(migraphx::make_op("add"), sfta, fpa);
        auto ra  = mm->add_instruction(migraphx::make_op("round"), msa);
        auto ca  = create_clip_op(p, 127.0f, -128.0f, ra);
        auto qa  = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            ca);

        auto q_dot = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), qa, qa);
        auto f_dot = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            q_dot);
        std::vector<float> v_alpha(f_dot->get_shape().elements(), 20.0f);
        auto new_alpha = mm->add_literal(migraphx::literal{f_dot->get_shape(), v_alpha});
        auto alpha_ab  = mm->add_instruction(migraphx::make_op("mul"), new_alpha, f_dot);
        mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int32_type)}}),
            alpha_ab);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{1.0f, 1.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
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
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fa = mm->add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vfa));
        auto conv_a = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pa);
        auto ma = mm->add_instruction(migraphx::make_op("mul"), fa, conv_a);

        // add the shift
        std::vector<float> vsa(sa.elements(), 1.0f);
        auto sfta =
            mm->add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vsa));
        auto msa = mm->add_instruction(migraphx::make_op("add"), sfta, ma);
        auto ra  = mm->add_instruction(migraphx::make_op("round"), msa);
        auto ca  = create_clip_op(p, 127.0f, -128.0f, ra);
        auto qa  = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = mm->add_literal(migraphx::literal({migraphx::shape::float_type, sb.lens()}, vfb));
        auto conv_b = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pb);
        auto mb = mm->insert_instruction(insert_loc, migraphx::make_op("mul"), fb, conv_b);
        auto rb = mm->insert_instruction(insert_loc, migraphx::make_op("round"), mb);
        auto cb = create_clip_op(insert_loc, p, 127.0f, -128.0f, rb);
        auto qb = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cb);

        auto qdot = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), qa, qb);
        auto fr = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            qdot);
        std::vector<float> v_alpha(fr->get_shape().elements(), 20.0f);
        auto new_alpha = mm->add_literal(migraphx::literal(fr->get_shape(), v_alpha));
        auto alpha_ab  = mm->add_instruction(migraphx::make_op("mul"), new_alpha, fr);
        auto fc        = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pc);
        std::vector<float> v_beta(fc->get_shape().elements(), 5.5f);
        auto beta   = mm->add_literal(migraphx::literal(fc->get_shape(), v_beta));
        auto beta_c = mm->add_instruction(migraphx::make_op("mul"), beta, fc);
        auto f_res  = mm->add_instruction(migraphx::make_op("add"), alpha_ab, beta_c);
        auto res    = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int32_type)}}),
            f_res);
        if(add_return)
        {
            mm->add_return({res});
        }

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    auto qp = create_int8_quantized_prog();
    EXPECT(p == qp);

    auto p_ret = create_program(true);
    migraphx::quantize_int8_impl(p_ret, quant_params, {"dot"});
    auto qp_ret = create_int8_quantized_prog(true);
    EXPECT(p_ret == qp_ret);
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
        mm->add_instruction(migraphx::make_op("dot", {{"alpha", 2.0f}, {"beta", 5.5f}}), fpa, pb);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::int8_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = mm->add_literal(migraphx::literal({migraphx::shape::float_type, sb.lens()}, vfb));
        auto mb = mm->insert_instruction(insert_loc, migraphx::make_op("mul"), fb, pb);
        auto rb = mm->insert_instruction(insert_loc, migraphx::make_op("round"), mb);
        auto cb = create_clip_op(insert_loc, p, 127.0f, -128.0f, rb);
        auto qb = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cb);

        auto qdot = mm->add_instruction(
            migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), pa, qb);
        auto fr = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            qdot);
        std::vector<float> v_alpha(fr->get_shape().elements(), 10.0f);
        auto new_alpha = mm->add_literal(migraphx::literal(fr->get_shape(), v_alpha));
        mm->add_instruction(migraphx::make_op("mul"), new_alpha, fr);

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
        mm->add_instruction(migraphx::make_op("convolution"), input, weights);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sx{migraphx::shape::float_type, {4, 3, 3, 3}};
        migraphx::shape sw{migraphx::shape::float_type, {4, 3, 3, 3}};
        auto px = mm->add_parameter("x", sx);
        auto pw = mm->add_parameter("w", sw);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfx(sx.elements(), 0.1f);
        auto fx = mm->add_literal(migraphx::literal(sx, vfx));
        auto mx = mm->add_instruction(migraphx::make_op("mul"), fx, px);
        auto rx = mm->add_instruction(migraphx::make_op("round"), mx);
        auto cx = create_clip_op(p, 127.0f, -128.0f, rx);
        auto qx = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cx);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pw);
        std::vector<float> vfw(sw.elements(), 0.1f);
        auto fw = mm->add_literal(migraphx::literal(sw, vfw));
        auto mw = mm->insert_instruction(insert_loc, migraphx::make_op("mul"), fw, pw);
        auto rw = mm->insert_instruction(insert_loc, migraphx::make_op("round"), mw);
        auto cw = create_clip_op(insert_loc, p, 127.0f, -128.0f, rw);
        auto qw = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cw);

        auto q_conv = mm->add_instruction(migraphx::make_op("quant_convolution"), qx, qw);
        auto f_conv = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            q_conv);
        std::vector<float> v_adj(f_conv->get_shape().elements(), 100.0f);
        auto adj = mm->add_literal(migraphx::literal(f_conv->get_shape(), v_adj));
        mm->add_instruction(migraphx::make_op("mul"), adj, f_conv);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"convolution"});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
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
        // quantize parameter a to int8 type, multiply the scale
        auto fpx = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            px);
        std::vector<float> vfx(sx.elements(), 0.1f);
        auto fx = mm->add_literal(migraphx::literal(fpx->get_shape(), vfx));
        auto mx = mm->add_instruction(migraphx::make_op("mul"), fx, fpx);
        auto rx = mm->add_instruction(migraphx::make_op("round"), mx);
        auto cx = create_clip_op(p, 127.0f, -128.0f, rx);
        auto qx = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cx);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pw);
        auto fpw        = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pw);
        std::vector<float> vfw(sw.elements(), 0.1f);
        auto fw = mm->add_literal(migraphx::literal(fpw->get_shape(), vfw));
        auto mw = mm->insert_instruction(insert_loc, migraphx::make_op("mul"), fw, fpw);
        auto rw = mm->insert_instruction(insert_loc, migraphx::make_op("round"), mw);
        auto cw = create_clip_op(insert_loc, p, 127.0f, -128.0f, rw);
        auto qw = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cw);

        auto q_conv = mm->add_instruction(migraphx::make_op("quant_convolution"), qx, qw);
        std::vector<float> v_adj(q_conv->get_shape().elements(), 100.0f);
        auto adj = mm->add_literal(migraphx::literal(q_conv->get_shape(), v_adj));
        mm->add_instruction(migraphx::make_op("mul"), q_conv, adj);

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
        // quantize parameter a to int8 type, multiply the scale
        auto fpx = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            px);
        std::vector<float> vfx(sx.elements(), 0.1f);
        auto fx = mm->add_literal(migraphx::literal(fpx->get_shape(), vfx));
        auto mx = mm->add_instruction(migraphx::make_op("mul"), fx, fpx);
        auto rx = mm->add_instruction(migraphx::make_op("round"), mx);
        auto cx = create_clip_op(p, 127.0f, -128.0f, rx);
        auto qx = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cx);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pw);
        auto fpw        = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            pw);
        std::vector<float> vfw(sw.elements(), 0.1f);
        auto fw = mm->add_literal(migraphx::literal(fpw->get_shape(), vfw));
        auto mw = mm->insert_instruction(insert_loc, migraphx::make_op("mul"), fw, fpw);
        auto rw = mm->insert_instruction(insert_loc, migraphx::make_op("round"), mw);
        auto cw = create_clip_op(insert_loc, p, 127.0f, -128.0f, rw);
        auto qw = mm->insert_instruction(
            insert_loc,
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::int8_type)}}),
            cw);

        auto q_conv = mm->add_instruction(migraphx::make_op("quant_convolution"), qx, qw);
        auto f_conv = mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
            q_conv);
        std::vector<float> v_adj(f_conv->get_shape().elements(), 100.0f);
        auto adj   = mm->add_literal(migraphx::literal(f_conv->get_shape(), v_adj));
        auto f_res = mm->add_instruction(migraphx::make_op("mul"), adj, f_conv);
        mm->add_instruction(
            migraphx::make_op("convert",
                              {{"target_type", migraphx::to_value(migraphx::shape::half_type)}}),
            f_res);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"convolution"});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
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
        mm->add_instruction(migraphx::make_op("dot"), pa, pb, pc);

        return p;
    };

    {
        auto p = create_program();
        migraphx::parameter_map m;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        m["a"] = migraphx::generate_argument(sa);
        m["c"] = migraphx::generate_argument(sc);
        std::vector<float> quant_result;
        migraphx::target ref_t = migraphx::ref::target{};
        run_prog(p, ref_t, m, quant_result, true);

        std::vector<float> no_quant_result;
        run_prog(p, ref_t, m, no_quant_result);

        EXPECT(migraphx::verify_range(quant_result, no_quant_result));
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
