/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_rope_qk_norm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <pointwise.hpp>
#include <reduce.hpp>
#include <test.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p,
                         {migraphx::gpu::fuse_rope_qk_norm{}, migraphx::dead_code_elimination{}});
}

static bool contains_op(const migraphx::module& m, const std::string& name)
{
    return std::any_of(m.begin(), m.end(), [&](const auto& ins) { return ins.name() == name; });
}

// Build the rmsnorm fused_reduce as fuse_pointwise_reduce produces it for the
// qwen3 q/k head norms
static migraphx::instruction_ref add_rmsnorm(migraphx::program& p,
                                             const std::string& name,
                                             migraphx::instruction_ref data,
                                             migraphx::instruction_ref weight)
{
    auto lens  = data->get_shape().lens();
    auto scale = 1.0f / lens.back();
    return add_reduce(
        p, name, {data, weight}, {3}, [&](auto* rm, const auto& inputs, const auto& axes) {
            auto sq =
                add_pointwise(p, rm, name + ":square", {inputs[0]}, [&](auto* pm, const auto& xs) {
                    auto slit = pm->add_literal(
                        migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {scale}});
                    auto cvt = pm->add_instruction(
                        migraphx::make_op("convert",
                                          {{"target_type", migraphx::shape::float_type}}),
                        xs[0]);
                    auto mul = pm->add_instruction(migraphx::make_op("mul"), cvt, cvt);
                    return pm->add_instruction(migraphx::make_op("mul"), mul, slit);
                });
            auto rsum = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sq);
            auto rsqrt =
                add_pointwise(p, rm, name + ":rsqrt", {rsum}, [&](auto* pm, const auto& xs) {
                    auto eps = pm->add_literal(
                        migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1e-6f}});
                    auto add = pm->add_instruction(migraphx::make_op("add"), xs[0], eps);
                    auto rs  = pm->add_instruction(migraphx::make_op("rsqrt"), add);
                    return pm->add_instruction(
                        migraphx::make_op("convert", {{"target_type", migraphx::shape::bf16_type}}),
                        rs);
                });
            auto mb = rm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                                          rsqrt);
            return add_pointwise(
                p, rm, name + ":apply", {inputs[0], mb, inputs[1]}, [&](auto* pm, const auto& xs) {
                    auto mul = pm->add_instruction(migraphx::make_op("mul"), xs[0], xs[1]);
                    return pm->add_instruction(migraphx::make_op("mul"), mul, xs[2]);
                });
        });
}

TEST_CASE(fuse_rope_qk_norm_qwen_pattern)
{
    const int64_t b = 2, nq = 4, nk = 2, d = 8;
    const int64_t h = nq + nk, total = nq + 2 * nk, w = total * d;
    migraphx::shape qkv_s{migraphx::shape::bf16_type, {std::size_t(b), 1, std::size_t(w)}};
    migraphx::shape w_s{migraphx::shape::bf16_type, {std::size_t(d)}};
    migraphx::shape cs_s{migraphx::shape::bf16_type, {std::size_t(b), 1, 1, std::size_t(d)}};

    migraphx::program p;
    {
        auto* mm  = p.get_main_module();
        auto qkv  = mm->add_parameter("qkv", qkv_s);
        auto qw   = mm->add_literal(migraphx::generate_literal(w_s, 0));
        auto kw   = mm->add_literal(migraphx::generate_literal(w_s, 1));
        auto cos  = mm->add_parameter("cos", cs_s);
        auto ssin = mm->add_parameter("ssin", cs_s);

        auto qsl = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {nq * d}}}), qkv);
        auto qrs =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {b, 1, nq, d}}}), qsl);
        auto qwb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 3}, {"out_lens", {b, 1, nq, d}}}), qw);
        auto frq = add_rmsnorm(p, "qnorm", qrs, qwb);
        auto qr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {b, 1, nq * d}}}), frq);

        auto ksl = mm->add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {2}}, {"starts", {nq * d}}, {"ends", {(nq + nk) * d}}}),
            qkv);
        auto krs =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {b, 1, nk, d}}}), ksl);
        auto kwb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 3}, {"out_lens", {b, 1, nk, d}}}), kw);
        auto frk = add_rmsnorm(p, "knorm", krs, kwb);
        auto kr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {b, 1, nk * d}}}), frk);

        auto vs = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {(nq + nk) * d}}, {"ends", {w}}}),
            qkv);
        auto cc = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), qr, kr, vs);
        auto rsh =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {b, 1, total, d}}}), cc);
        auto t = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), rsh);

        auto qk = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {h}}}), t);
        auto s2 = mm->add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {1, 3}}, {"starts", {0, d / 2}}, {"ends", {h, d}}}),
            t);
        auto s1 = mm->add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {1, 3}}, {"starts", {0, 0}}, {"ends", {h, d / 2}}}),
            t);
        auto rot = mm->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), s2, s1);

        auto cosb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {b, h, 1, d}}}), cos);
        auto ssinb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {b, h, 1, d}}}), ssin);

        auto rope = add_pointwise(p, "rope", {rot, ssinb, qk, cosb}, [](auto* pm, const auto& xs) {
            auto m1 = pm->add_instruction(migraphx::make_op("mul"), xs[0], xs[1]);
            auto m2 = pm->add_instruction(migraphx::make_op("mul"), xs[2], xs[3]);
            return pm->add_instruction(migraphx::make_op("add"), m2, m1);
        });
        auto vt   = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {h}}, {"ends", {total}}}), t);
        mm->add_return({rope, vt});
    }
    run_pass(p);
    const auto* mm = p.get_main_module();
    EXPECT(contains_op(*mm, "gpu::rope_qk_norm"));
    EXPECT(not contains_op(*mm, "concat"));
    EXPECT(not contains_op(*mm, "fused_reduce"));
    EXPECT(not contains_op(*mm, "pointwise"));

    auto fused = std::find_if(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "gpu::rope_qk_norm"; });
    auto v = fused->get_operator().to_value();
    EXPECT(v.at("num_heads").to<std::size_t>() == std::size_t(nq));
    EXPECT(migraphx::float_equal(v.at("eps").to<float>(), 1e-6f));
    EXPECT(migraphx::float_equal(v.at("ss_scale").to<float>(), 1.0f / d));
    EXPECT(fused->get_shape().lens() ==
           std::vector<std::size_t>{std::size_t(b), std::size_t(h), 1, std::size_t(d)});
}

// Prefill (sequence length > 1) must not fuse: the kernel only handles decode
TEST_CASE(fuse_rope_qk_norm_skips_non_decode)
{
    const int64_t b = 2, nq = 4, nk = 2, d = 8;
    const int64_t h = nq + nk;
    migraphx::shape in_s{migraphx::shape::bf16_type,
                         {std::size_t(b), std::size_t(h), 3, std::size_t(d)}};

    migraphx::program p;
    {
        auto* mm = p.get_main_module();
        auto x0  = mm->add_parameter("x0", in_s);
        auto x1  = mm->add_parameter("x1", in_s);
        auto x2  = mm->add_parameter("x2", in_s);
        auto x3  = mm->add_parameter("x3", in_s);
        auto cc  = mm->add_instruction(
            migraphx::make_op("concat", {{"axis", 3}}),
            mm->add_instruction(
                migraphx::make_op("slice", {{"axes", {3}}, {"starts", {d / 2}}, {"ends", {d}}}),
                x0),
            mm->add_instruction(
                migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {d / 2}}}),
                x0));
        auto rope = add_pointwise(p, "rope", {cc, x1, x2, x3}, [](auto* pm, const auto& xs) {
            auto m1 = pm->add_instruction(migraphx::make_op("mul"), xs[0], xs[1]);
            auto m2 = pm->add_instruction(migraphx::make_op("mul"), xs[2], xs[3]);
            return pm->add_instruction(migraphx::make_op("add"), m2, m1);
        });
        mm->add_return({rope});
    }
    run_pass(p);
    EXPECT(not contains_op(*p.get_main_module(), "gpu::rope_qk_norm"));
}

// Build the q-and-k-are-separate spelling of the fusion as
// fuse_pointwise_reduce leaves it: a standalone reduction for the reciprocal
// root and a two-output pointwise combining the halves of the row.
static migraphx::instruction_ref add_separate_rope(migraphx::program& p, bool swap_halves)
{
    const int64_t b = 2, n = 4, d = 8;
    const int64_t half = d / 2;
    const std::vector<std::size_t> lens{std::size_t(b), std::size_t(n), std::size_t(d)};
    const std::vector<std::size_t> half_lens{std::size_t(b), std::size_t(n), std::size_t(half)};

    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::bf16_type, lens});
    auto w = mm->add_parameter("w", migraphx::shape{migraphx::shape::bf16_type, {std::size_t(d)}});
    migraphx::shape angle_s{migraphx::shape::bf16_type, {std::size_t(b), 1, std::size_t(half)}};
    auto cos = mm->add_parameter("cos", angle_s);
    auto sin = mm->add_parameter("sin", angle_s);

    auto cvt = add_pointwise(p, "convert", {x}, [](auto* pm, const auto& xs) {
        return pm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), xs[0]);
    });
    auto rs = add_reduce(p, "rms", {cvt}, {2}, [&](auto* rm, const auto& inputs, const auto& axes) {
        auto sq   = add_pointwise(p, rm, "rms:square", {inputs[0]}, [&](auto* pm, const auto& xs) {
            auto slit = pm->add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1.0f / d}});
            auto mul = pm->add_instruction(migraphx::make_op("mul"), xs[0], xs[0]);
            return pm->add_instruction(migraphx::make_op("mul"), mul, slit);
        });
        auto rsum = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sq);
        return add_pointwise(p, rm, "rms:rsqrt", {rsum}, [&](auto* pm, const auto& xs) {
            auto eps = pm->add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1e-6f}});
            auto add = pm->add_instruction(migraphx::make_op("add"), xs[0], eps);
            return pm->add_instruction(migraphx::make_op("rsqrt"), add);
        });
    });

    auto broadcast = [&](migraphx::instruction_ref ins, const std::vector<std::size_t>& out) {
        return mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out}}), ins);
    };
    auto rsb  = broadcast(rs, lens);
    auto wb   = broadcast(w, lens);
    auto cosb = broadcast(cos, half_lens);
    auto sinb = broadcast(sin, half_lens);

    auto slice = [&](migraphx::instruction_ref ins, int64_t start) {
        return mm->add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {2}}, {"starts", {start}}, {"ends", {start + half}}}),
            ins);
    };
    std::vector<migraphx::instruction_ref> args{slice(cvt, 0),
                                                slice(rsb, 0),
                                                slice(wb, 0),
                                                slice(cvt, half),
                                                slice(rsb, half),
                                                slice(wb, half),
                                                sinb,
                                                cosb};

    auto* pm = p.create_module("rope");
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    std::transform(args.begin(), args.end(), std::back_inserter(params), [&](auto arg) {
        return pm->add_parameter("x" + std::to_string(params.size()),
                                 migraphx::shape{arg->get_shape().type()});
    });
    auto normalized = [&](std::size_t data, std::size_t root, std::size_t weight) {
        auto scaled = pm->add_instruction(migraphx::make_op("mul"), params[data], params[root]);
        auto back   = pm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bf16_type}}), scaled);
        return pm->add_instruction(migraphx::make_op("mul"), back, params[weight]);
    };
    auto n_lo = normalized(0, 1, 2);
    auto n_hi = normalized(3, 4, 5);
    auto term = [&](migraphx::instruction_ref norm, std::size_t angle) {
        return pm->add_instruction(migraphx::make_op("mul"), norm, params[angle]);
    };
    auto lo = pm->add_instruction(migraphx::make_op("sub"), term(n_lo, 7), term(n_hi, 6));
    auto hi = pm->add_instruction(migraphx::make_op("add"), term(n_hi, 7), term(n_lo, 6));
    pm->add_return({hi, lo});

    auto pw    = mm->add_instruction(migraphx::make_op("pointwise"), args, {pm});
    auto e_hi  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pw);
    auto e_lo  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), pw);
    auto first = swap_halves ? e_hi : e_lo;
    auto last  = swap_halves ? e_lo : e_hi;
    return mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), first, last);
}

TEST_CASE(fuse_rope_norm_separate_qk)
{
    migraphx::program p;
    p.get_main_module()->add_return({add_separate_rope(p, false)});
    run_pass(p);
    EXPECT(contains_op(*p.get_main_module(), "gpu::rope_norm"));
    EXPECT(not contains_op(*p.get_main_module(), "concat"));
}

// The halves have to go back in the order the rotation produced them
TEST_CASE(fuse_rope_norm_skips_swapped_halves)
{
    migraphx::program p;
    p.get_main_module()->add_return({add_separate_rope(p, true)});
    run_pass(p);
    EXPECT(not contains_op(*p.get_main_module(), "gpu::rope_norm"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
