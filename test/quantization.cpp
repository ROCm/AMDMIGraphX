#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"
#include <migraphx/half.hpp>

TEST_CASE(param_add)
{
    auto create_program_float = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1 = p.add_parameter("x", s);
        auto p2 = p.add_parameter("y", s);
        p.add_instruction(migraphx::op::add{}, p1, p2);

        return p;
    };

    auto create_program_half = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = p.add_parameter("x", s);
        auto hp1 = p.insert_instruction(std::next(p1), migraphx::op::convert{}, p1);
        auto p2  = p.add_parameter("y", s);
        auto hp2 = p.insert_instruction(std::next(p2), migraphx::op::convert{}, p2);
        auto hs  = p.add_instruction(migraphx::op::add{}, hp1, hp2);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hs);

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
}

TEST_CASE(param_add_sub)
{
    auto create_program_float = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1   = p.add_parameter("x", s);
        auto p2   = p.add_parameter("y", s);
        auto sum  = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto diff = p.add_instruction(migraphx::op::sub{}, sum, p2);
        p.add_instruction(migraphx::op::add{}, diff, p1);

        return p;
    };

    auto create_program_half_add = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = p.add_parameter("x", s);
        auto hp1 = p.insert_instruction(
            std::next(p1), migraphx::op::convert{migraphx::shape::half_type}, p1);
        auto p2  = p.add_parameter("y", s);
        auto hp2 = p.insert_instruction(
            std::next(p2), migraphx::op::convert{migraphx::shape::half_type}, p2);
        auto hsum  = p.add_instruction(migraphx::op::add{}, hp1, hp2);
        auto sum   = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hsum);
        auto diff  = p.add_instruction(migraphx::op::sub{}, sum, p2);
        auto hdiff = p.add_instruction(
            migraphx::op::convert{migraphx::op::convert{migraphx::shape::half_type}}, diff);
        auto res = p.add_instruction(migraphx::op::add{}, hdiff, hp1);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, res);

        return p;
    };

    auto create_program_half_sub = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = p.add_parameter("x", s);
        auto p2  = p.add_parameter("y", s);
        auto hp2 = p.insert_instruction(
            std::next(p2), migraphx::op::convert{migraphx::shape::half_type}, p2);
        auto sum   = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto hsum  = p.add_instruction(migraphx::op::convert{migraphx::shape::half_type}, sum);
        auto hdiff = p.add_instruction(migraphx::op::sub{}, hsum, hp2);
        auto diff  = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hdiff);
        p.add_instruction(migraphx::op::add{}, diff, p1);

        return p;
    };

    auto create_program_half_all = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = p.add_parameter("x", s);
        auto hp1 = p.insert_instruction(
            std::next(p1), migraphx::op::convert{migraphx::shape::half_type}, p1);
        auto p2  = p.add_parameter("y", s);
        auto hp2 = p.insert_instruction(
            std::next(p2), migraphx::op::convert{migraphx::shape::half_type}, p2);
        auto hsum  = p.add_instruction(migraphx::op::add{}, hp1, hp2);
        auto hdiff = p.add_instruction(migraphx::op::sub{}, hsum, hp2);
        auto hres  = p.add_instruction(migraphx::op::add{}, hdiff, hp1);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hres);

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
        migraphx::run_passes(p1, {migraphx::dead_code_elimination{}});

        EXPECT(p1 == p2);
    }
}

TEST_CASE(literal_add)
{
    auto create_program_float = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = p.add_literal(migraphx::literal(s, data));
        auto l2 = p.add_literal(migraphx::literal(s, data));
        p.add_instruction(migraphx::op::add{}, l1, l2);

        return p;
    };

    auto create_program_half = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::half_type, {2, 3}};
        std::vector<migraphx::half> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = p.add_literal(migraphx::literal(s, data));
        auto l2 = p.add_literal(migraphx::literal(s, data));
        auto hs = p.add_instruction(migraphx::op::add{}, l1, l2);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hs);

        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1, {"all"});
        migraphx::run_passes(p1,
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        migraphx::run_passes(p2,
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});

        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize_fp16(p1, {"add"});
        migraphx::run_passes(p1,
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        migraphx::run_passes(p2,
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
        migraphx::shape s1{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};

        auto p1 = p.add_parameter("x", s1);
        auto p2 = p.add_parameter("y", s1);
        auto pb = p.add_parameter("b", s2);
        auto pc = p.add_parameter("c", s2);
        auto pa = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto ps = p.add_instruction(migraphx::op::dot{}, pa, pb, pc);
        p.add_instruction(migraphx::op::dot{}, pa, ps);

        return p;
    };

    auto create_program_op = [&] {
        migraphx::program p;
        migraphx::shape s1{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};

        auto p1  = p.add_parameter("x", s1);
        auto p2  = p.add_parameter("y", s1);
        auto pb  = p.add_parameter("b", s2);
        auto pc  = p.add_parameter("c", s2);
        auto pa  = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto opb = p.insert_instruction(std::next(pb), migraphx::op::capture{1, test_func}, pb);
        auto opc = p.insert_instruction(std::next(pc), migraphx::op::capture{2, test_func}, pc);
        auto opa = p.add_instruction(migraphx::op::capture{0, test_func}, pa);
        auto ps  = p.add_instruction(migraphx::op::dot{}, opa, opb, opc);
        auto ops = p.add_instruction(migraphx::op::capture{3, test_func}, ps);
        p.add_instruction(migraphx::op::dot{}, opa, ops);

        return p;
    };

    {
        auto p             = create_program_float();
        auto op_capture_p  = create_program_op();
        migraphx::target t = migraphx::cpu::target{};
        migraphx::capture_arguments(p, t, {"dot", "convolution"});
        EXPECT(p == op_capture_p);
    }
}

TEST_CASE(dot_float)
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);

        p.add_instruction(migraphx::op::dot{2.0f, 1.5f}, pa, pb, pc);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fa = p.add_literal(migraphx::literal(sa, vfa));
        auto ma = p.add_instruction(migraphx::op::mul{}, fa, pa);
        auto ra = p.add_instruction(migraphx::op::round{}, ma);
        auto ca = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, ra);
        auto qa = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = p.add_literal(migraphx::literal(sb, vfb));
        auto mb = p.insert_instruction(insert_loc, migraphx::op::mul{}, fb, pb);
        auto rb = p.insert_instruction(insert_loc, migraphx::op::round{}, mb);
        auto cb = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rb);
        auto qb =
            p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cb);

        auto qdot = p.add_instruction(migraphx::op::quant_dot{1, 0}, qa, qb);
        auto fdot = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, qdot);
        std::vector<float> v_alpha(fdot->get_shape().elements(), 200.0f);
        auto new_alpha = p.add_literal(migraphx::literal(fdot->get_shape(), v_alpha));
        auto alpha_ab  = p.add_instruction(migraphx::op::mul{}, new_alpha, fdot);
        std::vector<float> v_beta(pc->get_shape().elements(), 1.5f);
        auto beta   = p.add_literal(migraphx::literal(pc->get_shape(), v_beta));
        auto beta_c = p.add_instruction(migraphx::op::mul{}, beta, pc);
        p.add_instruction(migraphx::op::add{}, alpha_ab, beta_c);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 0.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    migraphx::run_passes(p, {migraphx::dead_code_elimination{}});

    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(dot_double_2args)
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);

        p.add_instruction(migraphx::op::dot{2.0f, 1.5f}, pa, pb);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::double_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fpa = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, pa);
        auto fa  = p.add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vfa));
        auto ma  = p.add_instruction(migraphx::op::mul{}, fa, fpa);
        auto ra  = p.add_instruction(migraphx::op::round{}, ma);
        auto ca  = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, ra);
        auto qa  = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        auto fpb        = p.insert_instruction(
            insert_loc, migraphx::op::convert{migraphx::shape::float_type}, pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = p.add_literal(migraphx::literal({migraphx::shape::float_type, sb.lens()}, vfb));
        auto mb = p.insert_instruction(insert_loc, migraphx::op::mul{}, fb, fpb);
        auto rb = p.insert_instruction(insert_loc, migraphx::op::round{}, mb);
        auto cb = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rb);
        auto qb =
            p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cb);

        auto qdot = p.add_instruction(migraphx::op::quant_dot{1, 0}, qa, qb);
        auto fdot = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, qdot);
        std::vector<float> v_alpha(fdot->get_shape().elements(), 200.0f);
        auto new_alpha = p.add_literal(migraphx::literal(fdot->get_shape(), v_alpha));
        auto alpha_ab  = p.add_instruction(migraphx::op::mul{}, new_alpha, fdot);
        p.add_instruction(migraphx::op::convert{migraphx::shape::double_type}, alpha_ab);

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
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);

        p.add_instruction(migraphx::op::dot{20.0f, 50.5f}, pa, pb, pc);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fa = p.add_literal(migraphx::literal(sa, vfa));
        auto ma = p.add_instruction(migraphx::op::mul{}, fa, pa);
        // add the shift
        std::vector<float> vsa(sa.elements(), 1.0f);
        auto sfta = p.add_literal(migraphx::literal(sa, vsa));
        auto msa  = p.add_instruction(migraphx::op::add{}, sfta, ma);
        auto ra   = p.add_instruction(migraphx::op::round{}, msa);
        auto ca   = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, ra);
        auto qa   = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = p.add_literal(migraphx::literal(sb, vfb));
        auto mb = p.insert_instruction(insert_loc, migraphx::op::mul{}, fb, pb);
        auto rb = p.insert_instruction(insert_loc, migraphx::op::round{}, mb);
        auto cb = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rb);
        auto qb =
            p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cb);

        // quantize parameter c to int32 type
        auto qc = p.insert_instruction(
            std::next(pc), migraphx::op::convert{migraphx::shape::int32_type}, pc);

        auto qdot = p.add_instruction(migraphx::op::quant_dot{2000, 51}, qa, qb, qc);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, qdot);

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
        migraphx::shape sa{migraphx::shape::int32_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::int32_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::int32_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);

        p.add_instruction(migraphx::op::dot{20.0f, 50.0f}, pa, pb, pc);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::int32_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::int32_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::int32_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fa = p.add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vfa));
        auto conv_a = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, pa);
        auto ma     = p.add_instruction(migraphx::op::mul{}, fa, conv_a);

        // add the shift
        std::vector<float> vsa(sa.elements(), 1.0f);
        auto sfta = p.add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vsa));
        auto msa  = p.add_instruction(migraphx::op::add{}, sfta, ma);
        auto ra   = p.add_instruction(migraphx::op::round{}, msa);
        auto ca   = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, ra);
        auto qa   = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = p.add_literal(migraphx::literal({migraphx::shape::float_type, sb.lens()}, vfb));
        auto conv_b = p.insert_instruction(
            insert_loc, migraphx::op::convert{migraphx::shape::float_type}, pb);
        auto mb = p.insert_instruction(insert_loc, migraphx::op::mul{}, fb, conv_b);
        auto rb = p.insert_instruction(insert_loc, migraphx::op::round{}, mb);
        auto cb = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rb);
        auto qb =
            p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cb);

        p.add_instruction(migraphx::op::quant_dot{2000, 50}, qa, qb, pc);

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
        migraphx::shape s{migraphx::shape::int32_type, {16, 16}};
        auto pa = p.add_parameter("a", s);

        p.add_instruction(migraphx::op::dot{20.0f, 50.0f}, pa, pa);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::int32_type, {16, 16}};
        auto pa = p.add_parameter("a", s);

        // add the shift
        auto fpa = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, pa);
        std::vector<float> vsa(s.elements(), 1.0f);
        auto sfta = p.add_literal(migraphx::literal({migraphx::shape::float_type, s.lens()}, vsa));
        auto msa  = p.add_instruction(migraphx::op::add{}, sfta, fpa);
        auto ra   = p.add_instruction(migraphx::op::round{}, msa);
        auto ca   = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, ra);
        auto qa   = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, ca);

        auto q_dot = p.add_instruction(migraphx::op::quant_dot{1, 0}, qa, qa);
        auto f_dot = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, q_dot);
        std::vector<float> v_alpha(f_dot->get_shape().elements(), 20.0f);
        auto new_alpha = p.add_literal(migraphx::literal{f_dot->get_shape(), v_alpha});
        auto alpha_ab  = p.add_instruction(migraphx::op::mul{}, new_alpha, f_dot);
        p.add_instruction(migraphx::op::convert{migraphx::shape::int32_type}, alpha_ab);

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
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::int32_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::int32_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::int32_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);

        p.add_instruction(migraphx::op::dot{2.0f, 5.5f}, pa, pb, pc);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::int32_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::int32_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::int32_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fa = p.add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vfa));
        auto conv_a = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, pa);
        auto ma     = p.add_instruction(migraphx::op::mul{}, fa, conv_a);

        // add the shift
        std::vector<float> vsa(sa.elements(), 1.0f);
        auto sfta = p.add_literal(migraphx::literal({migraphx::shape::float_type, sa.lens()}, vsa));
        auto msa  = p.add_instruction(migraphx::op::add{}, sfta, ma);
        auto ra   = p.add_instruction(migraphx::op::round{}, msa);
        auto ca   = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, ra);
        auto qa   = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = p.add_literal(migraphx::literal({migraphx::shape::float_type, sb.lens()}, vfb));
        auto conv_b = p.insert_instruction(
            insert_loc, migraphx::op::convert{migraphx::shape::float_type}, pb);
        auto mb = p.insert_instruction(insert_loc, migraphx::op::mul{}, fb, conv_b);
        auto rb = p.insert_instruction(insert_loc, migraphx::op::round{}, mb);
        auto cb = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rb);
        auto qb =
            p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cb);

        auto qdot = p.add_instruction(migraphx::op::quant_dot{1, 0}, qa, qb);
        auto fr   = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, qdot);
        std::vector<float> v_alpha(fr->get_shape().elements(), 20.0f);
        auto new_alpha = p.add_literal(migraphx::literal(fr->get_shape(), v_alpha));
        auto alpha_ab  = p.add_instruction(migraphx::op::mul{}, new_alpha, fr);
        auto fc        = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, pc);
        std::vector<float> v_beta(fc->get_shape().elements(), 5.5f);
        auto beta   = p.add_literal(migraphx::literal(fc->get_shape(), v_beta));
        auto beta_c = p.add_instruction(migraphx::op::mul{}, beta, fc);
        auto f_res  = p.add_instruction(migraphx::op::add{}, alpha_ab, beta_c);
        p.add_instruction(migraphx::op::convert{migraphx::shape::int32_type}, f_res);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(dot_float_convert)
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::int8_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);

        auto fpa = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, pa);
        p.add_instruction(migraphx::op::dot{2.0f, 5.5f}, fpa, pb);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::int8_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = p.add_literal(migraphx::literal({migraphx::shape::float_type, sb.lens()}, vfb));
        auto mb = p.insert_instruction(insert_loc, migraphx::op::mul{}, fb, pb);
        auto rb = p.insert_instruction(insert_loc, migraphx::op::round{}, mb);
        auto cb = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rb);
        auto qb =
            p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cb);

        auto qdot = p.add_instruction(migraphx::op::quant_dot{1, 0}, pa, qb);
        auto fr   = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, qdot);
        std::vector<float> v_alpha(fr->get_shape().elements(), 10.0f);
        auto new_alpha = p.add_literal(migraphx::literal(fr->get_shape(), v_alpha));
        p.add_instruction(migraphx::op::mul{}, new_alpha, fr);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 1.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8_impl(p, quant_params, {"dot"});
    migraphx::run_passes(p, {migraphx::dead_code_elimination{}});
    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

TEST_CASE(conv_float)
{
    auto create_program = [] {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::convolution{}, input, weights);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sx{migraphx::shape::float_type, {4, 3, 3, 3}};
        migraphx::shape sw{migraphx::shape::float_type, {4, 3, 3, 3}};
        auto px = p.add_parameter("x", sx);
        auto pw = p.add_parameter("w", sw);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfx(sx.elements(), 0.1f);
        auto fx = p.add_literal(migraphx::literal(sx, vfx));
        auto mx = p.add_instruction(migraphx::op::mul{}, fx, px);
        auto rx = p.add_instruction(migraphx::op::round{}, mx);
        auto cx = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, rx);
        auto qx = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, cx);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pw);
        std::vector<float> vfw(sw.elements(), 0.1f);
        auto fw = p.add_literal(migraphx::literal(sw, vfw));
        auto mw = p.insert_instruction(insert_loc, migraphx::op::mul{}, fw, pw);
        auto rw = p.insert_instruction(insert_loc, migraphx::op::round{}, mw);
        auto cw = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rw);
        auto qw =
            p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cw);

        auto q_conv = p.add_instruction(migraphx::op::quant_convolution{}, qx, qw);
        auto f_conv = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, q_conv);
        std::vector<float> v_adj(f_conv->get_shape().elements(), 100.0f);
        auto adj = p.add_literal(migraphx::literal(f_conv->get_shape(), v_adj));
        p.add_instruction(migraphx::op::mul{}, adj, f_conv);

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
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::int32_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::int32_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::convolution{}, input, weights);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sx{migraphx::shape::int32_type, {4, 3, 3, 3}};
        migraphx::shape sw{migraphx::shape::int32_type, {4, 3, 3, 3}};
        auto px = p.add_parameter("x", sx);
        auto pw = p.add_parameter("w", sw);
        // quantize parameter a to int8 type, multiply the scale
        auto fpx = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, px);
        std::vector<float> vfx(sx.elements(), 0.1f);
        auto fx = p.add_literal(migraphx::literal(fpx->get_shape(), vfx));
        auto mx = p.add_instruction(migraphx::op::mul{}, fx, fpx);
        auto rx = p.add_instruction(migraphx::op::round{}, mx);
        auto cx = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, rx);
        auto qx = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, cx);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pw);
        auto fpw        = p.insert_instruction(
            insert_loc, migraphx::op::convert{migraphx::shape::float_type}, pw);
        std::vector<float> vfw(sw.elements(), 0.1f);
        auto fw = p.add_literal(migraphx::literal(fpw->get_shape(), vfw));
        auto mw = p.insert_instruction(insert_loc, migraphx::op::mul{}, fw, fpw);
        auto rw = p.insert_instruction(insert_loc, migraphx::op::round{}, mw);
        auto cw = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rw);
        auto qw =
            p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cw);

        auto q_conv = p.add_instruction(migraphx::op::quant_convolution{}, qx, qw);
        std::vector<float> v_adj(q_conv->get_shape().elements(), 100.0f);
        auto adj = p.add_literal(migraphx::literal(q_conv->get_shape(), v_adj));
        p.add_instruction(migraphx::op::mul{}, q_conv, adj);

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
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::convolution{}, input, weights);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sx{migraphx::shape::half_type, {4, 3, 3, 3}};
        migraphx::shape sw{migraphx::shape::half_type, {4, 3, 3, 3}};
        auto px = p.add_parameter("x", sx);
        auto pw = p.add_parameter("w", sw);
        // quantize parameter a to int8 type, multiply the scale
        auto fpx = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, px);
        std::vector<float> vfx(sx.elements(), 0.1f);
        auto fx = p.add_literal(migraphx::literal(fpx->get_shape(), vfx));
        auto mx = p.add_instruction(migraphx::op::mul{}, fx, fpx);
        auto rx = p.add_instruction(migraphx::op::round{}, mx);
        auto cx = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, rx);
        auto qx = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, cx);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pw);
        auto fpw        = p.insert_instruction(
            insert_loc, migraphx::op::convert{migraphx::shape::float_type}, pw);
        std::vector<float> vfw(sw.elements(), 0.1f);
        auto fw = p.add_literal(migraphx::literal(fpw->get_shape(), vfw));
        auto mw = p.insert_instruction(insert_loc, migraphx::op::mul{}, fw, fpw);
        auto rw = p.insert_instruction(insert_loc, migraphx::op::round{}, mw);
        auto cw = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rw);
        auto qw =
            p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cw);

        auto q_conv = p.add_instruction(migraphx::op::quant_convolution{}, qx, qw);
        auto f_conv = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, q_conv);
        std::vector<float> v_adj(f_conv->get_shape().elements(), 100.0f);
        auto adj   = p.add_literal(migraphx::literal(f_conv->get_shape(), v_adj));
        auto f_res = p.add_instruction(migraphx::op::mul{}, adj, f_conv);
        p.add_instruction(migraphx::op::convert{migraphx::shape::half_type}, f_res);

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
                       migraphx::program::parameter_map& m_in,
                       std::vector<float>& res) {
        p.compile(t);
        migraphx::program::parameter_map m;
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
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto p1 = p.add_parameter("x", s);
        auto p2 = p.add_parameter("y", s);
        p.add_instruction(migraphx::op::add{}, p1, p2);

        return p;
    };

    {
        auto p = create_program();
        migraphx::program::parameter_map m;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        m["x"] = migraphx::generate_argument(s);
        std::vector<float> cpu_result;
        migraphx::target cpu_t = migraphx::cpu::target{};
        run_prog(p, cpu_t, m, cpu_result);

        std::vector<float> orig_result;
        run_prog(p, cpu_t, m, orig_result);

        EXPECT(migraphx::verify_range(cpu_result, orig_result));
    }
}

TEST_CASE(int8_quantization_dot)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::program::parameter_map& m_in,
                       std::vector<float>& res,
                       bool b_quantize = false) {
        if(b_quantize)
        {
            std::vector<migraphx::program::parameter_map> cali_data;
            cali_data.push_back(m_in);
            migraphx::quantize_int8(p, t, cali_data);
        }
        p.compile(t);
        migraphx::program::parameter_map m;
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
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);
        p.add_instruction(migraphx::op::dot{}, pa, pb, pc);

        return p;
    };

    {
        auto p = create_program();
        migraphx::program::parameter_map m;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        m["a"] = migraphx::generate_argument(sa);
        m["c"] = migraphx::generate_argument(sc);
        std::vector<float> quant_result;
        migraphx::target cpu_t = migraphx::cpu::target{};
        run_prog(p, cpu_t, m, quant_result, true);

        std::vector<float> no_quant_result;
        run_prog(p, cpu_t, m, no_quant_result);

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
            std::vector<migraphx::program::parameter_map> cali_data;
            migraphx::quantize_int8(p, t, cali_data);
        }
        p.compile(t);
        migraphx::program::parameter_map m;

        auto result = t.copy_from(p.eval(m).back());
        result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
    };

    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sx{migraphx::shape::float_type, {4, 2, 2, 2}};
        migraphx::shape sw{migraphx::shape::float_type, {4, 2, 2, 2}};
        std::vector<float> v(sx.elements(), 0.5f);
        auto input   = p.add_literal(migraphx::literal(sx, v));
        auto weights = p.add_literal(migraphx::literal(sw, v));
        p.add_instruction(migraphx::op::convolution{}, input, weights);

        return p;
    };

    {
        auto p = create_program();
        std::vector<float> quant_result;
        migraphx::target cpu_t = migraphx::cpu::target{};
        run_prog(p, cpu_t, quant_result, true);

        std::vector<float> no_quant_result;
        run_prog(p, cpu_t, no_quant_result);

        EXPECT(migraphx::verify_range(quant_result, no_quant_result));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
