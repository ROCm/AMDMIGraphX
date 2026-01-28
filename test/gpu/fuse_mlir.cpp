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
#include <migraphx/generate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/param_utils.hpp>
#include <basic_ops.hpp>
#include <group.hpp>
#include <test.hpp>
#include <pointwise.hpp>
#include <reduce.hpp>
#include <utility>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_USE_SPECIFIC_OPS);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MLIR_GEG_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_CEG_FUSION);

struct non_mlir_op
{
    std::string name() const { return "non_mlir_op"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(1);
        return inputs.at(0);
    }
};

static void run_pass(migraphx::program& p, migraphx::gpu::fuse_mlir fm = {})
{
    static migraphx::gpu::context ctx;
    fm.ctx          = &ctx;
    fm.enable_extra = true;
    migraphx::run_passes(p, {fm, migraphx::dead_code_elimination{}});
}

template <class F>
static migraphx::instruction_ref add_mlir(migraphx::program& p,
                                          const std::string& name,
                                          std::vector<migraphx::instruction_ref> inputs,
                                          std::vector<std::string> arg_names,
                                          const F& f)
{
    assert(inputs.size() == arg_names.size() and "One interior parameter name given per input.");
    auto* mm = p.get_main_module();
    auto* pm = p.create_module(name);
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    for(size_t i = 0, e = inputs.size(); i < e; ++i)
    {
        params.push_back(pm->add_parameter(arg_names[i], inputs[i]->get_shape().as_standard()));
    }
    auto values = f(pm, params);
    auto root   = std::get<0>(values);
    auto r      = std::get<1>(values);
    auto_add_return(pm, r);
    return mm->add_instruction(
        migraphx::make_op("gpu::mlir_op", {{"op", migraphx::to_value(root)}}), inputs, {pm});
}

template <class F>
static migraphx::instruction_ref add_mlir(migraphx::program& p,
                                          const std::string& name,
                                          std::vector<migraphx::instruction_ref> inputs,
                                          const F& f)
{
    std::vector<std::string> arg_names;
    migraphx::transform(migraphx::range(inputs.size()), std::back_inserter(arg_names), [&](auto i) {
        return migraphx::param_name(i);
    });
    return add_mlir(p, name, std::move(inputs), std::move(arg_names), std::move(f));
}

TEST_CASE(dot_reshapes_add)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 3}});
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
        auto dot_sq = mm->add_instruction(migraphx::make_op("squeeze"), dot_trans);
        auto add    = add_pointwise(p1, "main:pointwise0", {dot_sq, x}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 3}});
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto dot_trans = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
                auto dot_rsp = pm->add_instruction(migraphx::make_op("squeeze"), dot_trans);
                auto add     = pm->add_instruction(migraphx::make_op("add"), dot_rsp, inputs[2]);
                return std::make_tuple(dot->get_operator(), add);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_add)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add = add_pointwise(p1, "main:pointwise0", {dot, x}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot, inputs[2]);
                return std::make_tuple(dot->get_operator(), add);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_transpose_reshape_add)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 6, 6}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 6}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s1);
        auto x   = mm->add_parameter("x", s1);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto xtranspose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
        auto xreshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s1.lens()}}), xtranspose);
        auto add = add_pointwise(p1, "main:pointwise0", {dot, xreshape}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s1);
        auto x   = mm->add_parameter("x", s1);
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto xtranspose = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), inputs[2]);
                auto xreshape = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", s1.lens()}}), xtranspose);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot, xreshape);
                return std::make_tuple(dot->get_operator(), add);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_reshape_lazy_add)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 6, 6}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 36}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s1);
        auto x   = mm->add_parameter("x", s2);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto xreshape_lazy =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", s1.lens()}}), x);
        auto add =
            add_pointwise(p1, "main:pointwise0", {dot, xreshape_lazy}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s1);
        auto x   = mm->add_parameter("x", s2);
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto xreshape_lazy = pm->add_instruction(
                    migraphx::make_op("reshape_lazy", {{"dims", s1.lens()}}), inputs[2]);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot, xreshape_lazy);
                return std::make_tuple(dot->get_operator(), add);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_backwards)
{
    migraphx::shape os{migraphx::shape::float_type, {{1, 1, 5, 5}}};
    migraphx::shape is{migraphx::shape::float_type, {1, 1, 3, 3}};
    migraphx::shape ws{migraphx::shape::float_type, {1, 1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", is);
        auto w       = mm->add_parameter("w", ws);
        auto conv_bk = mm->add_instruction(migraphx::make_op("convolution_backwards"), x, w);
        mm->add_return({conv_bk});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", is);
        auto w   = mm->add_parameter("w", ws);
        auto conv_bk =
            add_mlir(p2,
                     "mlir_convolution_backwards0",
                     {x, w},
                     {"y0", "y1"},
                     [=](auto* pm, const auto& inputs) {
                         auto c = pm->add_instruction(
                             migraphx::make_op("convolution_backwards"), inputs[0], inputs[1]);
                         return std::make_tuple(c->get_operator(), c);
                     });
        mm->add_return({conv_bk});
    }

    std::string opt = migraphx::string_value_of(MIGRAPHX_MLIR_USE_SPECIFIC_OPS{}, "");
    if(opt.find("convolution_backwards") != std::string::npos)
        EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_broadcast_mul)
{
    migraphx::shape os{migraphx::shape::float_type, {4, 56, 122, 122}};
    migraphx::shape is{migraphx::shape::float_type, {4, 14, 1, 1}};
    migraphx::shape ws{migraphx::shape::float_type, {56, 14, 1, 1}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", is);
        auto y     = mm->add_parameter("y", os);
        auto w     = mm->add_parameter("w", ws);
        auto conv  = mm->add_instruction(migraphx::make_op("convolution"), x, w);
        auto convb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", os.lens()}}), conv);
        auto mul = add_pointwise(p1, "main:pointwise0", {convb, y}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", is);
        auto y    = mm->add_parameter("y", os);
        auto w    = mm->add_parameter("w", ws);
        auto conv = add_mlir(
            p2, "mlir_convolution0", {x, w}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto c =
                    pm->add_instruction(migraphx::make_op("convolution"), inputs[0], inputs[1]);
                return std::make_tuple(c->get_operator(), c);
            });
        auto convb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", os.lens()}}), conv);
        auto mul = add_pointwise(p2, "main:pointwise0", {convb, y}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(multi_use_dot_trans_add_pooling_sub)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 1, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto dot       = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot_trans = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), dot);
        auto add = add_pointwise(p1, "main:pointwise0", {dot_trans, x}, single_pointwise("add"));
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {0, 0, 0, 1}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                add);
        auto sub = add_pointwise(p1, "main:pointwise1", {dot, pooling}, single_pointwise("sub"));
        mm->add_return({sub});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto dot_trans = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), dot);

                auto add = pm->add_instruction(migraphx::make_op("add"), dot_trans, inputs[2]);
                return std::make_tuple(dot->get_operator(),
                                       std::vector<migraphx::instruction_ref>{add, dot});
            });
        auto fused_dot_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {0, 0, 0, 1}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                fused_dot_add);
        auto dot = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto sub = add_pointwise(p2, "main:pointwise1", {dot, pooling}, single_pointwise("sub"));
        mm->add_return({sub});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_multi_use_trans_add_pooling_sub)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 5, 5}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
        auto dot_unsq =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 5, 4}}}), dot_trans);
        auto add = add_pointwise(p1, "main:pointwise0", {dot_unsq, x}, single_pointwise("add"));
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {1, 0, 0, 0}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                add);
        auto sub =
            add_pointwise(p1, "main:pointwise1", {dot_unsq, pooling}, single_pointwise("sub"));
        mm->add_return({sub});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto dot_trans = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
                auto dot_unsq = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {1, 1, 5, 4}}}), dot_trans);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot_unsq, inputs[2]);
                return std::make_tuple(dot->get_operator(),
                                       std::vector<migraphx::instruction_ref>{add, dot});
            });
        auto fused_dot_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {1, 0, 0, 0}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                fused_dot_add);
        auto dot = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto dot_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
        auto dot_reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 5, 4}}}), dot_trans);
        auto sub =
            add_pointwise(p2, "main:pointwise1", {dot_reshape, pooling}, single_pointwise("sub"));
        mm->add_return({sub});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_dot_pointwise)
// this test does not run GEG fusion since it is so small
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 5, 5}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s2);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), dot1, c);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, dot2}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto c   = mm->add_parameter("c", s2);
        auto dot1 =
            add_mlir(p2, "mlir_dot0", {a, b}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        auto dot2 =
            add_mlir(p2, "mlir_dot1", {dot1, c}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        auto add = add_pointwise(p2, "main:pointwise0", {dot1, dot2}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(dot_dot_pointwise_pointwise)
// this test does not run GEG fusion since it is so small
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 5, 5}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s2);
        auto x    = mm->add_parameter("d", s1);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), dot1, c);
        auto add1 = add_pointwise(p1, "main:pointwise0", {dot2, x}, single_pointwise("add"));
        auto add2 = add_pointwise(p1, "main:pointwise1", {dot1, add1}, single_pointwise("add"));
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto c   = mm->add_parameter("c", s2);
        auto x   = mm->add_parameter("d", s1);
        auto dot1 =
            add_mlir(p2, "mlir_dot0", {a, b}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {dot1, c, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot, inputs[2]);
                return std::make_tuple(dot->get_operator(), add);
            });
        auto add2 = add_pointwise(p2, "main:pointwise1", {dot1, fused}, single_pointwise("add"));
        mm->add_return({add2});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(add_dot)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto add = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), add, b);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto fused =
            add_mlir(p2,
                     "main:pointwise0:mlir_dot0",
                     {x, y, b},
                     {"x0", "x1", "x2"},
                     [=](auto* pm, const auto& inputs) {
                         auto add =
                             pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                         auto dot = pm->add_instruction(migraphx::make_op("dot"), add, inputs[2]);
                         return std::make_tuple(dot->get_operator(), dot);
                     });
        mm->add_return({fused});
    }
    if(not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(relu_dot)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto b    = mm->add_parameter("b", s);
        auto x    = mm->add_parameter("x", s);
        auto relu = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("relu"));
        auto dot  = mm->add_instruction(migraphx::make_op("dot"), relu, b);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto fused =
            add_mlir(p2,
                     "main:pointwise0:mlir_dot0",
                     {x, b},
                     {"x0", "x1"},
                     [=](auto* pm, const auto& inputs) {
                         auto relu = pm->add_instruction(migraphx::make_op("relu"), inputs[0]);
                         auto dot  = pm->add_instruction(migraphx::make_op("dot"), relu, inputs[1]);
                         return std::make_tuple(dot->get_operator(), dot);
                     });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(relu_relu_dot)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto relux = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("relu"));
        auto reluy = add_pointwise(p1, "main:pointwise1", {y}, single_pointwise("relu"));
        auto dot   = mm->add_instruction(migraphx::make_op("dot"), relux, reluy);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto fused =
            add_mlir(p2,
                     "main:pointwise0:main:pointwise1:mlir_dot0",
                     {x, y},
                     {"x0", "x1"},
                     [=](auto* pm, const auto& inputs) {
                         auto relux = pm->add_instruction(migraphx::make_op("relu"), inputs[0]);
                         auto reluy = pm->add_instruction(migraphx::make_op("relu"), inputs[1]);
                         auto dot   = pm->add_instruction(migraphx::make_op("dot"), relux, reluy);
                         return std::make_tuple(dot->get_operator(), dot);
                     });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dequantizelinear_dot)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();

        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
        auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::int8_type, {2, 5, 2}});
        auto scalelit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 2, 2}}));

        auto unsqueeze1 =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scalelit);
        auto broadcast1 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
        auto reshape1 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
        auto scale = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

        auto unsqueeze2 =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), zplit);
        auto broadcast2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
        auto reshape2 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
        auto zp = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

        auto dq = add_pointwise(
            p1, "main:pointwise0", {y, scale, zp}, single_pointwise("dequantizelinear"));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, dq);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
        auto y   = mm->add_parameter("y", migraphx::shape{migraphx::shape::int8_type, {2, 5, 2}});
        auto scalelit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 2, 2}}));

        auto fused = add_mlir(
            p2,
            "main:pointwise0:mlir_dot0",
            {y, scalelit, zplit, x},
            {"x0", "x1", "x2", "x3"},
            [=](auto* pm, const auto& inputs) {
                auto unsqueeze1 =
                    pm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), inputs[1]);
                auto broadcast1 = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
                auto reshape1 = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
                auto scale = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}),
                    reshape1);

                auto unsqueeze2 =
                    pm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), inputs[2]);
                auto broadcast2 = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
                auto reshape2 = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
                auto zp = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}),
                    reshape2);

                auto dq = pm->add_instruction(
                    migraphx::make_op("dequantizelinear"), inputs[0], scale, zp);
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[3], dq);
                return std::make_tuple(dot->get_operator(), dot);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unsigned_dequantizelinear_dot)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();

        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
        auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::uint8_type, {2, 5, 2}});
        auto scalelit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::uint8_type, {2, 2, 2}}));

        auto unsqueeze1 =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scalelit);
        auto broadcast1 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
        auto reshape1 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
        auto scale = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

        auto unsqueeze2 =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), zplit);
        auto broadcast2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
        auto reshape2 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
        auto zp = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

        auto dq = add_pointwise(
            p1, "main:pointwise0", {y, scale, zp}, single_pointwise("dequantizelinear"));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, dq);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
        auto y   = mm->add_parameter("y", migraphx::shape{migraphx::shape::uint8_type, {2, 5, 2}});
        auto scalelit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::uint8_type, {2, 2, 2}}));

        auto fused = add_mlir(
            p2,
            "main:pointwise0:mlir_dot0",
            {y, scalelit, zplit, x},
            {"x0", "x1", "x2", "x3"},
            [=](auto* pm, const auto& inputs) {
                auto unsqueeze1 =
                    pm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), inputs[1]);
                auto broadcast1 = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
                auto reshape1 = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
                auto scale = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}),
                    reshape1);

                auto unsqueeze2 =
                    pm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), inputs[2]);
                auto broadcast2 = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
                auto reshape2 = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
                auto zp = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}),
                    reshape2);

                auto dq = pm->add_instruction(
                    migraphx::make_op("dequantizelinear"), inputs[0], scale, zp);
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[3], dq);
                return std::make_tuple(dot->get_operator(), dot);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unpack_int4_dot)
{
    migraphx::program p1;
    {
        auto* m   = p1.get_main_module();
        auto x    = m->add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 4}});
        auto pk_w = m->add_parameter("wt_int4", {migraphx::shape::int8_type, {1, 8, 4, 2}});
        auto w    = m->add_instruction(migraphx::make_op("unpack_int4"), pk_w);
        auto dot  = m->add_instruction(migraphx::make_op("quant_dot"), x, w); // w: {1,8,4,4}
        m->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* m   = p2.get_main_module();
        auto x    = m->add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 4}});
        auto pk_w = m->add_parameter("wt_int4", {migraphx::shape::int8_type, {1, 8, 4, 2}});

        auto fused = add_mlir(
            p2, "int4:mlir_quant_dot0", {x, pk_w}, {"x1", "x2"}, [=](auto* pm, const auto& inputs) {
                auto unpk_w = pm->add_instruction(migraphx::make_op("unpack_int4"), inputs[1]);
                auto q = pm->add_instruction(migraphx::make_op("quant_dot"), inputs[0], unpk_w);
                return std::make_tuple(q->get_operator(), q);
            });
        m->add_return({fused});
    }

    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unpack_int4_dot_2)
{
    migraphx::program p1;
    {
        auto* m = p1.get_main_module();

        auto pk_x = m->add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 2}});
        auto x    = m->add_instruction(migraphx::make_op("unpack_int4"), pk_x); // {1,8,4,4}

        auto pk_w = m->add_parameter("wt_int4", {migraphx::shape::int8_type, {1, 8, 4, 2}});
        auto w    = m->add_instruction(migraphx::make_op("unpack_int4"), pk_w); // {1,8,4,4}

        auto dot = m->add_instruction(migraphx::make_op("quant_dot"), x, w);
        m->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* m   = p2.get_main_module();
        auto x    = m->add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 2}});
        auto pk_w = m->add_parameter("wt_int4", {migraphx::shape::int8_type, {1, 8, 4, 2}});

        auto fused = add_mlir(
            p2, "int4:mlir_quant_dot0", {x, pk_w}, {"x1", "x2"}, [=](auto* pm, const auto& inputs) {
                auto unpk_x = pm->add_instruction(migraphx::make_op("unpack_int4"), inputs[0]);
                auto unpk_w = pm->add_instruction(migraphx::make_op("unpack_int4"), inputs[1]);
                auto q      = pm->add_instruction(migraphx::make_op("quant_dot"), unpk_x, unpk_w);
                return std::make_tuple(q->get_operator(), q);
            });
        m->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(int_quant_dot_abs)
{
    migraphx::shape s_a{migraphx::shape::int8_type, {5, 4}};
    migraphx::shape s_b{migraphx::shape::int8_type, {4, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s_a);
        auto b   = mm->add_parameter("b", s_b);
        auto dot = mm->add_instruction(migraphx::make_op("quant_dot"), a, b);
        auto abs = add_pointwise(p1, "main:pointwise0", {dot}, single_pointwise("abs"));
        mm->add_return({abs});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s_a);
        auto b   = mm->add_parameter("b", s_b);
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b}, [=](auto* pm, const auto& inputs) {
                auto dot =
                    pm->add_instruction(migraphx::make_op("quant_dot"), inputs[0], inputs[1]);
                auto abs = pm->add_instruction(migraphx::make_op("abs"), dot);
                return std::make_tuple(dot->get_operator(), abs);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(int_quant_dot_tanh_fails)
{
    migraphx::shape s_a{migraphx::shape::int8_type, {5, 4}};
    migraphx::shape s_b{migraphx::shape::int8_type, {4, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s_a);
        auto b    = mm->add_parameter("b", s_b);
        auto dot  = mm->add_instruction(migraphx::make_op("quant_dot"), a, b);
        auto tanh = add_pointwise(p1, "main:pointwise0", {dot}, single_pointwise("tanh"));
        mm->add_return({tanh});
    }
    // This pass should not fuse as int32_t tanh isn't supported.
    run_pass(p1);
    auto* mm = p1.get_main_module();
    bool has_pointwise =
        std::any_of(mm->begin(), mm->end(), [&](const auto& i) { return i.name() == "pointwise"; });
    EXPECT(has_pointwise);
}

TEST_CASE(conv_split_reduce)
{
    migraphx::shape s_x{migraphx::shape::float_type, {2, 4, 64, 64}};
    migraphx::shape s_w{migraphx::shape::float_type, {320, 4, 3, 3}};
    migraphx::shape s_b{migraphx::shape::float_type, {32}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w   = mm->add_parameter("w", s_w);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto mb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}), b);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        auto reshape = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
        auto add = add_pointwise(p1, "main:pointwise0", {reshape, mb}, single_pointwise("add"));
        auto mean_var = add_reduce(
            p1,
            "main:split_reduce0",
            {add},
            {2, 3, 4},
            "assign_add",
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto xx    = add_pointwise(p1, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                return {rsum2, rsum1};
            });
        auto var =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), mean_var);
        auto mean =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), mean_var);
        mm->add_return({var, mean});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w   = mm->add_parameter("w", s_w);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_main:split_reduce0",
                     {x, w, b},
                     {"x0", "x1", "x2"},
                     [=](auto* pm, const auto& inputs) {
                         auto conv = pm->add_instruction(
                             migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                             inputs[0],
                             inputs[1]);
                         auto reshape = pm->add_instruction(
                             migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
                         auto mb = pm->add_instruction(
                             migraphx::make_op("broadcast",
                                               {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}),
                             inputs[2]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), reshape, mb);
                         auto mul  = pm->add_instruction(migraphx::make_op("mul"), add, add);
                         auto mean = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), add);
                         auto var = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), mul);
                         return std::make_tuple(
                             migraphx::make_op("gpu::mlir_op",
                                               {{"op", migraphx::to_value(conv->get_operator())}}),
                             std::vector<migraphx::instruction_ref>{var, mean});
                     });
        auto mean = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto var  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        mm->add_return({var, mean});
    }
    if(not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_add_split_reduce_multi_use)
{
    migraphx::shape s_x{migraphx::shape::float_type, {2, 4, 64, 64}};
    migraphx::shape s_w{migraphx::shape::float_type, {320, 4, 3, 3}};
    migraphx::shape s_b{migraphx::shape::float_type, {32}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w   = mm->add_parameter("w", s_w);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto mb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}), b);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        auto reshape = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
        auto add = add_pointwise(p1, "main:pointwise0", {reshape, mb}, single_pointwise("add"));
        auto mean_var = add_reduce(
            p1,
            "main:split_reduce0",
            {add},
            {2, 3, 4},
            "assign_add",
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto xx    = add_pointwise(p1, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                return {rsum2, rsum1};
            });
        auto var =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), mean_var);
        auto mean =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), mean_var);
        auto mean_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", add->get_shape().lens()}}), mean);
        auto var_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", add->get_shape().lens()}}), var);
        auto norm = add_pointwise(
            p1, "main:pointwise2", {add, mean_mb, var_mb}, [=](auto* pm, const auto& inputs) {
                auto sub =
                    pm->add_instruction(migraphx::make_op("sub"), inputs.at(0), inputs.at(1));
                return pm->add_instruction(migraphx::make_op("div"), sub, inputs.at(2));
            });
        mm->add_return({norm});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w   = mm->add_parameter("w", s_w);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_main:split_reduce0",
                     {x, w, b},
                     {"x0", "x1", "x2"},
                     [=](auto* pm, const auto& inputs) {
                         auto conv = pm->add_instruction(
                             migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                             inputs[0],
                             inputs[1]);
                         auto reshape = pm->add_instruction(
                             migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
                         auto mb = pm->add_instruction(
                             migraphx::make_op("broadcast",
                                               {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}),
                             inputs[2]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), reshape, mb);
                         auto mul  = pm->add_instruction(migraphx::make_op("mul"), add, add);
                         auto mean = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), add);
                         auto var = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), mul);
                         return std::make_tuple(
                             migraphx::make_op("gpu::mlir_op",
                                               {{"op", migraphx::to_value(conv->get_operator())}}),
                             std::vector<migraphx::instruction_ref>{var, mean, add});
                     });
        auto cba  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), fused);
        auto var  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto mean = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto mean_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", cba->get_shape().lens()}}), mean);
        auto var_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", cba->get_shape().lens()}}), var);
        auto norm = add_pointwise(
            p2, "main:pointwise2", {cba, mean_mb, var_mb}, [=](auto* pm, const auto& inputs) {
                auto sub =
                    pm->add_instruction(migraphx::make_op("sub"), inputs.at(0), inputs.at(1));
                return pm->add_instruction(migraphx::make_op("div"), sub, inputs.at(2));
            });
        mm->add_return({norm});
    }
    if(not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_add_split_reduce_multi_use_conv)
{
    migraphx::shape s_x{migraphx::shape::float_type, {2, 4, 64, 64}};
    migraphx::shape s_w1{migraphx::shape::float_type, {320, 4, 3, 3}};
    migraphx::shape s_w2{migraphx::shape::float_type, {320, 320, 3, 3}};
    migraphx::shape s_b{migraphx::shape::float_type, {32}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w1  = mm->add_parameter("w1", s_w1);
        auto w2  = mm->add_parameter("w2", s_w2);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto mb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}), b);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w1);
        auto reshape = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
        auto add = add_pointwise(p1, "main:pointwise0", {reshape, mb}, single_pointwise("add"));
        auto mean_var = add_reduce(
            p1,
            "main:split_reduce0",
            {add},
            {2, 3, 4},
            "assign_add",
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto xx    = add_pointwise(p1, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                return {rsum2, rsum1};
            });
        auto var =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), mean_var);
        auto mean =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), mean_var);
        auto mean_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", add->get_shape().lens()}}), mean);
        auto mean_rsp = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), mean_mb);
        auto var_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", add->get_shape().lens()}}), var);
        auto var_rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), var_mb);
        auto add_rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), add);
        auto norm = add_pointwise(
            p1, "main:pointwise2", {add_rsp, mean_rsp, var_rsp}, [=](auto* pm, const auto& inputs) {
                auto sub =
                    pm->add_instruction(migraphx::make_op("sub"), inputs.at(0), inputs.at(1));
                return pm->add_instruction(migraphx::make_op("div"), sub, inputs.at(2));
            });
        auto conv_2 = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), norm, w2);
        mm->add_return({conv_2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w1  = mm->add_parameter("w1", s_w1);
        auto w2  = mm->add_parameter("w2", s_w2);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_main:split_reduce0",
                     {x, w1, b},
                     {"x0", "x1", "x2"},
                     [=](auto* pm, const auto& inputs) {
                         auto conv = pm->add_instruction(
                             migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                             inputs[0],
                             inputs[1]);
                         auto reshape = pm->add_instruction(
                             migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
                         auto mb = pm->add_instruction(
                             migraphx::make_op("broadcast",
                                               {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}),
                             inputs[2]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), reshape, mb);
                         auto mul  = pm->add_instruction(migraphx::make_op("mul"), add, add);
                         auto mean = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), add);
                         auto var = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), mul);
                         return std::make_tuple(
                             migraphx::make_op("gpu::mlir_op",
                                               {{"op", migraphx::to_value(conv->get_operator())}}),
                             std::vector<migraphx::instruction_ref>{var, mean, add});
                     });
        auto cba  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), fused);
        auto var  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto mean = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto input_fused_conv = add_mlir(
            p2,
            "main:pointwise2:mlir_convolution1",
            {cba, mean, var, w2},
            {"x0", "x1", "x2", "x3"},
            [=](auto* pm, const auto& inputs) {
                auto mean_mb = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", cba->get_shape().lens()}}),
                    inputs.at(1));
                auto mean_rsp = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), mean_mb);
                auto var_mb = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", cba->get_shape().lens()}}),
                    inputs.at(2));
                auto var_rsp = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), var_mb);
                auto cba_rsp = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), inputs.at(0));
                auto sub  = pm->add_instruction(migraphx::make_op("sub"), cba_rsp, mean_rsp);
                auto div  = pm->add_instruction(migraphx::make_op("div"), sub, var_rsp);
                auto conv = pm->add_instruction(
                    migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                    div,
                    inputs.at(3));
                return std::make_tuple(conv->get_operator(), conv);
            });
        mm->add_return({input_fused_conv});
    }
    if(not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION{}) or
       not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(standalone_attention)
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};
    migraphx::shape s2{migraphx::shape::bool_type, {1, 12, 256, 256}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("1", s1);
        auto b   = mm->add_parameter("2", s1);
        auto b1  = mm->add_parameter("3", s1);
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);

        auto group = add_group(
            p1,
            "attn0",
            "attention",
            {a, b, b1},
            {"x0", "x1", "x2"},
            [=](auto* gm, const auto& inputs) {
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
                rmax = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                rsum = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[2]);
                return std::vector<migraphx::instruction_ref>{gemm2};
            });
        mm->add_return({group});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("1", s1);
        auto b     = mm->add_parameter("2", s1);
        auto b1    = mm->add_parameter("3", s1);
        auto fused = add_mlir(
            p2, "mlir_attn0", {a, b, b1}, {"x0", "x1", "x2"}, [=](auto* pm, const auto& inputs) {
                auto fb = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[1]);
                auto fb1 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[2]);
                auto gemm1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], fb);
                auto rmax =
                    pm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
                rmax = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = pm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
                auto exp = pm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    pm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                rsum = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div = pm->add_instruction(migraphx::make_op("div"), exp, rsum);

                auto gemm2 = pm->add_instruction(migraphx::make_op("dot"), div, fb1);
                return std::make_tuple(gemm2->get_operator(), gemm2);
            });
        mm->add_return({fused});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(fused_attention)
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};
    migraphx::shape s2{migraphx::shape::bool_type, {1, 12, 256, 256}};
    auto s1_elements = s1.elements();
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        auto c      = mm->add_parameter("5", s1);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);

        auto group = add_group(
            p1, "attn0", "attention", {a, b, select, b1}, [=](auto* gm, const auto& inputs) {
                auto ten   = gm->add_literal(migraphx::literal{s1, tens});
                auto eight = gm->add_literal(migraphx::literal{s1, eights});
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto mul   = gm->add_instruction(migraphx::make_op("mul"), gemm1, eight);
                auto where = gm->add_instruction(migraphx::make_op("where"), inputs[2], mul, ten);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
                rmax = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                rsum = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[3]);
                return std::vector<migraphx::instruction_ref>{gemm2};
            });
        auto trailing_pw =
            add_pointwise(p1, mm, "main:pointwise0", {group, c}, single_pointwise("add"));
        mm->add_return({trailing_pw});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        auto c      = mm->add_parameter("5", s1);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        auto fused = add_mlir(
            p2,
            "mlir_attn0",
            {a, b, select, b1, c},
            {"x0", "x1", "x2", "x3", "x4"},
            [=](auto* pm, const auto& inputs) {
                auto ten   = pm->add_literal(migraphx::literal{s1, tens});
                auto eight = pm->add_literal(migraphx::literal{s1, eights});
                auto fb    = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[1]);
                auto fb1 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[3]);
                auto gemm1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], fb);
                auto mul   = pm->add_instruction(migraphx::make_op("mul"), gemm1, eight);
                auto where = pm->add_instruction(migraphx::make_op("where"), inputs[2], mul, ten);
                auto rmax =
                    pm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
                rmax = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = pm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
                auto exp = pm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    pm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                rsum = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div   = pm->add_instruction(migraphx::make_op("div"), exp, rsum);
                auto gemm2 = pm->add_instruction(migraphx::make_op("dot"), div, fb1);
                auto add   = pm->add_instruction(migraphx::make_op("add"), gemm2, inputs[4]);
                return std::make_tuple(gemm2->get_operator(), add);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(lse_attention)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 12, 256, 256}};
    migraphx::shape s2{migraphx::shape::bool_type, {1, 12, 256, 256}};
    auto s1_elements = s1.elements();
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);

        auto group = add_group(
            p1,
            "attn0",
            "attention",
            {a, b, eight, select, ten, b1},
            [=](auto* gm, const auto& inputs) {
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto mul   = gm->add_instruction(migraphx::make_op("mul"), gemm1, inputs[2]);
                auto where =
                    gm->add_instruction(migraphx::make_op("where"), inputs[3], mul, inputs[4]);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
                auto rmax_mb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), where, rmax_mb);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                auto rsum_mb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum_mb);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[5]);
                auto log   = gm->add_instruction(migraphx::make_op("log"), rsum);
                auto add   = gm->add_instruction(migraphx::make_op("add"), log, rmax);
                return std::vector<migraphx::instruction_ref>{gemm2, add};
            });

        auto adjust_lse =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), group);

        auto lse =
            add_pointwise(p1, "main:pointwise0", {adjust_lse}, [=](auto* pm, const auto& inputs) {
                auto log2   = pm->add_literal(1.44238f);
                auto log2se = pm->add_instruction(migraphx::make_op("mul"), inputs.at(0), log2);
                return pm->add_instruction(
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
                    log2se);
            });

        auto lse_squeeze = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), lse);
        auto gemm2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), group);

        mm->add_return({gemm2, lse_squeeze});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});

        auto fused = add_mlir(
            p2,
            "mlir_attn0",
            {a, b, eight, select, ten, b1},
            {"x0", "x1", "x2", "x3", "x4", "x5"},
            [=](auto* pm, const auto& inputs) {
                auto log2 = pm->add_literal(1.44238f);
                auto fb   = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[1]);
                auto fb1 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[5]);
                auto gemm1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], fb);
                auto mul   = pm->add_instruction(migraphx::make_op("mul"), gemm1, inputs[2]);
                auto where =
                    pm->add_instruction(migraphx::make_op("where"), inputs[3], mul, inputs[4]);
                auto rmax =
                    pm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
                auto rmax_mb = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = pm->add_instruction(migraphx::make_op("sub"), where, rmax_mb);
                auto exp = pm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    pm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                auto rsum_mb = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div   = pm->add_instruction(migraphx::make_op("div"), exp, rsum_mb);
                auto gemm2 = pm->add_instruction(migraphx::make_op("dot"), div, fb1);
                auto log   = pm->add_instruction(migraphx::make_op("log"), rsum);
                auto add   = pm->add_instruction(migraphx::make_op("add"), log, rmax);
                log2       = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", add->get_shape().lens()}}),
                    log2);
                auto log2se = pm->add_instruction(migraphx::make_op("mul"), add, log2);
                auto lse    = pm->add_instruction(
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
                    log2se);
                return std::make_tuple(gemm2->get_operator(),
                                       std::vector<migraphx::instruction_ref>{gemm2, lse});
            });

        auto lse = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto lse_squeeze = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), lse);
        auto gemm2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);

        mm->add_return({gemm2, lse_squeeze});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_output_reshapes)
{
    migraphx::shape s1 = migraphx::shape::from_permutation(
        migraphx::shape::half_type, {64, 64, 160, 160}, {0, 2, 3, 1});
    // equivalent: migraphx::shape s1{migraphx::shape::half_type, {64, 64, 160, 160}, {1638400, 1,
    // 10240, 64}};
    migraphx::shape s2{migraphx::shape::half_type, {64, 64, 1, 1}, {0, 1, 0, 0}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto a       = mm->add_parameter("x", s1);
        auto b       = mm->add_parameter("w", s2);
        auto c       = mm->add_parameter("b", s1);
        auto conv    = mm->add_instruction(migraphx::make_op("convolution"), a, b);
        auto add     = add_pointwise(p1, "main:pointwise0", {conv, c}, single_pointwise("add"));
        auto reshape = mm->add_instruction(
            migraphx::make_op("reshape_lazy", {{"dims", {64, 2, 32, 160, 160}}}), add);
        auto transpose_0 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0, 3, 4, 2}}}), reshape);
        auto contiguous  = mm->add_instruction(migraphx::make_op("contiguous"), transpose_0);
        auto transpose_1 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 3}}}), contiguous);
        auto slice_0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
            transpose_1);
        auto slice_1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
            transpose_1);
        mm->add_return({slice_0, slice_1});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto a       = mm->add_parameter("x", s1);
        auto b       = mm->add_parameter("w", s2);
        auto c       = mm->add_parameter("b", s1);
        auto mlir_op = add_mlir(
            p2,
            "mlir_main:pointwise0_reshape_lazy_transpose_contiguous_transpose",
            {a, b, c},
            {"x0", "x1", "x2"},
            [=](auto* pm, const auto& inputs) {
                auto conv =
                    pm->add_instruction(migraphx::make_op("convolution"), inputs[0], inputs[1]);
                auto add     = pm->add_instruction(migraphx::make_op("add"), conv, inputs[2]);
                auto reshape = pm->add_instruction(
                    migraphx::make_op("reshape_lazy", {{"dims", {64, 2, 32, 160, 160}}}), add);
                auto transpose_0 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {1, 0, 3, 4, 2}}}), reshape);
                auto contiguous = pm->add_instruction(migraphx::make_op("contiguous"), transpose_0);
                auto transpose_1 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 3}}}), contiguous);
                return std::make_tuple(transpose_1->get_operator(), transpose_1);
            });
        auto slice_0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), mlir_op);
        auto slice_1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), mlir_op);
        mm->add_return({slice_0, slice_1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(channel_slice_convolution)
{
    migraphx::shape s1 = migraphx::shape::from_permutation(
        migraphx::shape::half_type, {64, 64, 160, 160}, {0, 2, 3, 1});
    // equivalent: migraphx::shape s1{migraphx::shape::half_type, {64, 64, 160, 160}, {1638400, 1,
    // 10240, 64}};
    migraphx::shape s2{migraphx::shape::half_type, {64, 64, 1, 1}, {0, 1, 0, 0}};
    migraphx::shape s3{migraphx::shape::half_type, {32, 32, 1, 1}, {0, 1, 0, 0}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto a       = mm->add_parameter("x", s1);
        auto b       = mm->add_parameter("w", s2);
        auto c       = mm->add_parameter("b", s1);
        auto d       = mm->add_parameter("g", s3);
        auto conv    = mm->add_instruction(migraphx::make_op("convolution"), a, b);
        auto add     = add_pointwise(p1, "main:pointwise0", {conv, c}, single_pointwise("add"));
        auto slice_0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {32}}}), add);
        auto slice_1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {32}}, {"ends", {64}}}), add);
        auto conv_1 = mm->add_instruction(migraphx::make_op("convolution"), slice_0, d);
        auto conv_2 = mm->add_instruction(migraphx::make_op("convolution"), slice_1, d);
        mm->add_return({conv_1, conv_2});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto a       = mm->add_parameter("x", s1);
        auto b       = mm->add_parameter("w", s2);
        auto c       = mm->add_parameter("b", s1);
        auto d       = mm->add_parameter("g", s3);
        auto mlir_op = add_mlir(
            p2,
            "mlir_main:pointwise0_reshape_lazy_transpose_contiguous_transpose",
            {a, b, c},
            {"x0", "x1", "x2"},
            [=](auto* pm, const auto& inputs) {
                auto conv =
                    pm->add_instruction(migraphx::make_op("convolution"), inputs[0], inputs[1]);
                auto add     = pm->add_instruction(migraphx::make_op("add"), conv, inputs[2]);
                auto reshape = pm->add_instruction(
                    migraphx::make_op("reshape_lazy", {{"dims", {64, 2, 32, 160, 160}}}), add);
                auto transpose_0 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {1, 0, 3, 4, 2}}}), reshape);
                auto contiguous = pm->add_instruction(migraphx::make_op("contiguous"), transpose_0);
                auto transpose_1 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 3}}}), contiguous);
                return std::make_tuple(transpose_1->get_operator(), transpose_1);
            });

        auto identity = mm->add_instruction(migraphx::make_op("identity"), mlir_op);

        auto mlir_conv0 = add_mlir(
            p2,
            "mlir_convolution1",
            {identity, d},
            {"y0", "y1"},
            [=](auto* pm, const auto& inputs) {
                auto slice_0 = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                    inputs[0]);
                auto squeeze_0 =
                    pm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice_0);
                auto conv_0 =
                    pm->add_instruction(migraphx::make_op("convolution"), squeeze_0, inputs[1]);
                return std::make_tuple(conv_0->get_operator(), conv_0);
            });

        auto mlir_conv1 = add_mlir(
            p2,
            "mlir_convolution2",
            {identity, d},
            {"y0", "y1"},
            [=](auto* pm, const auto& inputs) {
                auto slice_1 = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
                    inputs[0]);
                auto squeeze_1 =
                    pm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice_1);
                auto conv_1 =
                    pm->add_instruction(migraphx::make_op("convolution"), squeeze_1, inputs[1]);
                return std::make_tuple(conv_1->get_operator(), conv_1);
            });

        mm->add_return({mlir_conv0, mlir_conv1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unpack_fp4_dot_even)
{
    migraphx::program p1;
    {
        auto* m       = p1.get_main_module();
        auto packed_a = m->add_parameter("a", {migraphx::shape::fp4x2_type, {1, 3, 8, 4}});
        auto packed_b = m->add_parameter("b", {migraphx::shape::fp4x2_type, {1, 3, 8, 4}});
        auto scale_a  = m->add_parameter("scale_a", {migraphx::shape::float_type, {1, 3, 8, 8}});
        auto scale_b  = m->add_parameter("scale_b", {migraphx::shape::float_type, {1, 3, 8, 8}});
        auto unpack_a = m->add_instruction(migraphx::make_op("unpack_fp4"), packed_a);
        auto unpack_b = m->add_instruction(migraphx::make_op("unpack_fp4"), packed_b);
        auto dot      = m->add_instruction(
            migraphx::make_op("quant_dot"), unpack_a, unpack_b, scale_a, scale_b);
        m->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* m       = p2.get_main_module();
        auto packed_a = m->add_parameter("a", {migraphx::shape::fp4x2_type, {1, 3, 8, 4}});
        auto packed_b = m->add_parameter("b", {migraphx::shape::fp4x2_type, {1, 3, 8, 4}});
        auto scale_a  = m->add_parameter("scale_a", {migraphx::shape::float_type, {1, 3, 8, 8}});
        auto scale_b  = m->add_parameter("scale_b", {migraphx::shape::float_type, {1, 3, 8, 8}});
        auto fused    = add_mlir(
            p2,
            "fp4:mlir_quant_dot0",
            {packed_a, packed_b, scale_a, scale_b},
            {"x1", "x2", "x3", "x4"},
            [=](auto* pm, const auto& inputs) {
                auto unpack_a = pm->add_instruction(migraphx::make_op("unpack_fp4"), inputs[0]);
                auto unpack_b = pm->add_instruction(migraphx::make_op("unpack_fp4"), inputs[1]);
                auto dot      = pm->add_instruction(
                    migraphx::make_op("quant_dot"), unpack_a, unpack_b, inputs[2], inputs[3]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        m->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unpack_fp4_dot_odd)
{
    migraphx::program p1;
    {
        auto* m       = p1.get_main_module();
        auto packed_a = m->add_parameter("a", {migraphx::shape::fp4x2_type, {1, 3, 7, 4}});
        auto packed_b = m->add_parameter("b", {migraphx::shape::fp4x2_type, {1, 3, 7, 4}});
        auto scale_a  = m->add_parameter("scale_a", {migraphx::shape::float_type, {1, 3, 7, 7}});
        auto scale_b  = m->add_parameter("scale_b", {migraphx::shape::float_type, {1, 3, 7, 7}});
        auto unpack_a = m->add_instruction(migraphx::make_op("unpack_fp4"), packed_a);
        auto unpack_b = m->add_instruction(migraphx::make_op("unpack_fp4"), packed_b);
        auto slice_a  = m->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {7}}}), unpack_a);
        auto slice_b = m->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {7}}}), unpack_b);
        auto dot =
            m->add_instruction(migraphx::make_op("quant_dot"), slice_a, slice_b, scale_a, scale_b);
        m->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* m       = p2.get_main_module();
        auto packed_a = m->add_parameter("a", {migraphx::shape::fp4x2_type, {1, 3, 7, 4}});
        auto packed_b = m->add_parameter("b", {migraphx::shape::fp4x2_type, {1, 3, 7, 4}});
        auto scale_a  = m->add_parameter("scale_a", {migraphx::shape::float_type, {1, 3, 7, 7}});
        auto scale_b  = m->add_parameter("scale_b", {migraphx::shape::float_type, {1, 3, 7, 7}});
        auto fused    = add_mlir(
            p2,
            "fp4:mlir_quant_dot0",
            {packed_a, packed_b, scale_a, scale_b},
            {"x1", "x2", "x3", "x4"},
            [=](auto* pm, const auto& inputs) {
                auto unpack_a = pm->add_instruction(migraphx::make_op("unpack_fp4"), inputs[0]);
                auto slice_a  = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {7}}}),
                    unpack_a);
                auto unpack_b = pm->add_instruction(migraphx::make_op("unpack_fp4"), inputs[1]);
                auto slice_b  = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {7}}}),
                    unpack_b);
                auto dot = pm->add_instruction(
                    migraphx::make_op("quant_dot"), slice_a, slice_b, inputs[2], inputs[3]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        m->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_add_dot)
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto x    = mm->add_parameter("x", s3);
        auto y    = mm->add_parameter("y", s4);
        auto dot1 = mm->add_instruction(
            migraphx::make_op("dot"), a, b); // {1024,4}, m + 1000 > avg (n, k, gemmO)
        auto add =
            add_pointwise(p1, "main:pointwise0", {dot1, x}, single_pointwise("add")); // {1024,4}
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), add, y);            // {1024,2}
        mm->add_return({dot2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto x     = mm->add_parameter("x", s3);
        auto y     = mm->add_parameter("y", s4);
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_mlir_dot1_geg",
                     {a, b, x, y},
                     [=](auto* pm, const auto& inputs) {
                         auto dot1 =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                         auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, inputs[3]);
                         return std::make_tuple(dot2->get_operator(), dot2);
                     });
        mm->add_return({fused});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_add_dot_abc_f32)
// MLIR currently only supports (A*B)*C GEG patterns
{
    migraphx::shape s1{migraphx::shape::float_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::float_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::float_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto x    = mm->add_parameter("x", s3);
        auto y    = mm->add_parameter("y", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b); // {1024,4}
        auto add =
            add_pointwise(p1, "main:pointwise0", {dot1, x}, single_pointwise("add")); // {1024, 4}
        auto dot2 =
            mm->add_instruction(migraphx::make_op("dot"), add, y); // {1024, 4}*{4, 2} = {1024, 2}
        mm->add_return({dot2});
    }
    run_pass(p1);
    // ensure "geg" is present. Earlier tests ensure the fusion is correct. This is just to ensure
    // it happens.
    std::stringstream ss;
    ss << p1;
    std::string program_str = ss.str();

    // regardless if the matmul is correctly oriented, f32 geg should not happen on navi
    auto device_name = migraphx::gpu::get_device_name();
    bool is_navi =
        migraphx::starts_with(device_name, "gfx11") or migraphx::starts_with(device_name, "gfx12");
    // fusion should not happen if the device is navi or the fusion flag is disabled
    if(is_navi or migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        EXPECT(program_str.find("geg") == std::string::npos);
    else
        EXPECT(program_str.find("geg") != std::string::npos);
}

TEST_CASE(dot_add_dot_abc_fp16)
// MLIR currently only supports (A*B)*C GEG patterns
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto x    = mm->add_parameter("x", s3);
        auto y    = mm->add_parameter("y", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b); // {1024,4}
        auto add =
            add_pointwise(p1, "main:pointwise0", {dot1, x}, single_pointwise("add")); // {1024, 4}
        auto dot2 =
            mm->add_instruction(migraphx::make_op("dot"), add, y); // {1024, 4}*{4, 2} = {1024, 2}
        mm->add_return({dot2});
    }
    run_pass(p1);
    std::stringstream ss;
    ss << p1;
    std::string program_str = ss.str();

    // ensure "geg" is present if the fusion flag is enabled; type is fp16 so it should
    // run regardless of if navi
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        EXPECT(program_str.find("geg") == std::string::npos);
    else
        EXPECT(program_str.find("geg") != std::string::npos);
}

TEST_CASE(dot_add_dot_cab)
// MLIR currently does not support C*(A*B) GEG patterns
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {2, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto x    = mm->add_parameter("x", s3);
        auto y    = mm->add_parameter("y", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b); // {2,4}
        auto add =
            add_pointwise(p1, "main:pointwise0", {dot1, x}, single_pointwise("add")); // {2, 4}
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), y, add); // {4, 2}*{2, 4} = {4, 4}
        mm->add_return({dot2});
    }
    run_pass(p1);
    // ensure "geg" is not in the fused program
    std::stringstream ss;
    ss << p1;
    std::string program_str = ss.str();
    EXPECT(program_str.find("geg") == std::string::npos);
}

TEST_CASE(dot_mul_dot)
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto x    = mm->add_parameter("x", s3);
        auto y    = mm->add_parameter("y", s4);
        auto dot1 = mm->add_instruction(
            migraphx::make_op("dot"), a, b); // {1024,4}, m + 1000 > avg (n, k, gemmO)
        auto mul =
            add_pointwise(p1, "main:pointwise0", {dot1, x}, single_pointwise("mul")); // {1024,4}
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), mul, y);            // {1024,2}
        mm->add_return({dot2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto x     = mm->add_parameter("x", s3);
        auto y     = mm->add_parameter("y", s4);
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_mlir_dot1_geg",
                     {a, b, x, y},
                     [=](auto* pm, const auto& inputs) {
                         auto dot1 =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                         auto mul  = pm->add_instruction(migraphx::make_op("mul"), dot1, inputs[2]);
                         auto dot2 = pm->add_instruction(migraphx::make_op("dot"), mul, inputs[3]);
                         return std::make_tuple(dot2->get_operator(), dot2);
                     });
        mm->add_return({fused});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_add)
{
    migraphx::shape is{migraphx::shape::float_type, {4, 14, 122, 122}};
    migraphx::shape ys{migraphx::shape::float_type, {4, 56, 122, 122}};
    migraphx::shape ws{migraphx::shape::float_type, {56, 14, 1, 1}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", is);
        auto y    = mm->add_parameter("y", ys);
        auto w    = mm->add_parameter("w", ws);
        auto conv = mm->add_instruction(migraphx::make_op("convolution"), x, w);
        auto add  = add_pointwise(p1, "main:pointwise0", {conv, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", is);
        auto y     = mm->add_parameter("y", ys);
        auto w     = mm->add_parameter("w", ws);
        auto fused = add_mlir(p2,
                              "mlir_main:pointwise0",
                              {x, w, y},
                              {"x0", "x1", "x2"},
                              [=](auto* pm, const auto& inputs) {
                                  auto c = pm->add_instruction(
                                      migraphx::make_op("convolution"), inputs[0], inputs[1]);
                                  auto add =
                                      pm->add_instruction(migraphx::make_op("add"), c, inputs[2]);
                                  return std::make_tuple(c->get_operator(), add);
                              });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_add_dot)
{
    migraphx::shape is{migraphx::shape::half_type, {2, 4, 8, 8}};
    migraphx::shape ys{migraphx::shape::half_type, {2, 8, 8, 8}};
    migraphx::shape ws{migraphx::shape::half_type, {8, 4, 1, 1}};
    migraphx::shape zs{migraphx::shape::half_type, {2, 8, 8, 4}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", is);
        auto y    = mm->add_parameter("y", ys);
        auto w    = mm->add_parameter("w", ws);
        auto z    = mm->add_parameter("z", zs);
        auto conv = mm->add_instruction(migraphx::make_op("convolution"), x, w);
        auto add  = add_pointwise(p1, "main:pointwise0", {conv, y}, single_pointwise("add"));
        auto dot  = mm->add_instruction(migraphx::make_op("dot"), add, z);

        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", is);
        auto y   = mm->add_parameter("y", ys);
        auto w   = mm->add_parameter("w", ws);
        auto z   = mm->add_parameter("z", zs);
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_geg",
                     {x, w, y, z},
                     {"x0", "x1", "x2", "x3"},
                     [=](auto* pm, const auto& inputs) {
                         auto c = pm->add_instruction(
                             migraphx::make_op("convolution"), inputs[0], inputs[1]);
                         auto add = pm->add_instruction(migraphx::make_op("add"), c, inputs[2]);
                         auto dot = pm->add_instruction(migraphx::make_op("dot"), add, inputs[3]);
                         return std::make_tuple(dot->get_operator(), dot);
                     });
        mm->add_return({fused});
    }
    if(not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_CEG_FUSION{}) or
       migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_multi_user_add)
// G ->optional R -> E fusion
// G has two users, one external to fusion
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s);
        auto b    = mm->add_parameter("b", s);
        auto c    = mm->add_parameter("c", s);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, c}, single_pointwise("add"));
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot1);
        mm->add_return({add, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto c   = mm->add_parameter("c", s);
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, c}, [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{add, dot1});
            });
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_dot =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), get_dot);
        mm->add_return({get_add, transpose});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_add_multi_user_dot, "GEG multi-output intermediates not supported")
// GEG fusion has two outputs, E has external user
// not currently supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s3);
        auto d    = mm->add_parameter("d", s4);
        auto dot1 =
            mm->add_instruction(migraphx::make_op("dot"), a, b); // {1024, 3} x {3, 4} = {1024, 4}
        auto add = add_pointwise(p1,
                                 "main:pointwise0",
                                 {dot1, c},
                                 single_pointwise("add")); // {1024, 4} + {1024, 4} = {1024, 4}
        auto dot2 =
            mm->add_instruction(migraphx::make_op("dot"), add, d); // {1024, 4} x {4, 2} = {1024, 2}
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2);
        mm->add_return({add, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto c     = mm->add_parameter("c", s3);
        auto d     = mm->add_parameter("d", s4);
        auto fused = add_mlir(
            p2, "mlir_main:pointwise0_geg", {a, b, c, d}, [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, inputs[3]);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, add});
            });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, transpose});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_add_multi_user_dot_with_transpose,
               "GEG multi-output intermediates not supported")
// GEG fusion has two outputs, E has external user
// not currently supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {2, 4}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s3);
        auto d    = mm->add_parameter("d", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, c}, single_pointwise("add"));
        auto d_t =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), d);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), add, d_t);
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2);
        mm->add_return({add, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto c     = mm->add_parameter("c", s3);
        auto d     = mm->add_parameter("d", s4);
        auto fused = add_mlir(
            p2, "mlir_main:pointwise0_geg", {a, b, c, d}, [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                auto d_t  = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {1, 0}}}), inputs[3]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, d_t);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, add});
            });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, transpose});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_add_multi_user_dot_two_externals, "GEG multi-output intermediates not supported")
// GEG fusion has two outputs, E has external user
// not currently supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s3);
        auto d    = mm->add_parameter("d", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, c}, single_pointwise("add"));
        auto external_t1 =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), d);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), add, d);
        auto external_t2 =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2);
        mm->add_return({add, external_t1, external_t2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto c     = mm->add_parameter("c", s3);
        auto d     = mm->add_parameter("d", s4);
        auto fused = add_mlir(
            p2, "mlir_main:pointwise0_geg", {a, b, c, d}, [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, inputs[3]);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, add});
            });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto external_t1 =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), d);
        auto external_t2 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, external_t1, external_t2});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_add_multi_user_dot_input_used_before,
               "GEG multi-output intermediates not supported")
// GEG fusion has two outputs, E has external user.
// Base case for testing inputs being defined within the span
// of will-be-fused ops
// This also shows the relu being fused, since it is a unary op
// currently not supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto c   = mm->add_parameter("c", s3);
        auto d   = mm->add_parameter("d", s4);

        auto external_relu = add_pointwise(p1, "main:pointwise1", {d}, single_pointwise("relu"));

        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, c}, single_pointwise("add"));
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), add, external_relu);
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2);
        mm->add_return({add, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto c     = mm->add_parameter("c", s3);
        auto d     = mm->add_parameter("d", s4);
        auto fused = add_mlir(
            p2,
            "main:pointwise1:mlir_main:pointwise0_geg",
            {d, a, b, c},
            [=](auto* pm, const auto& inputs) {
                auto external_relu = pm->add_instruction(migraphx::make_op("relu"), inputs[0]);
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[1], inputs[2]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[3]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, external_relu);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, add});
            });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto external_t = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, external_t});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_add_multi_user_dot_input_used_after,
               "GEG multi-output intermediates not supported")
// GEG fusion has two outputs, E has external user
// Testing inputs being defined within the span of will-be-fused ops
// This also shows the relu being fused, since it is a unary op.
// Result should be, and is, equivalent to the previous test
// currently not supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a             = mm->add_parameter("a", s1);
        auto b             = mm->add_parameter("b", s2);
        auto c             = mm->add_parameter("c", s3);
        auto d             = mm->add_parameter("d", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, c}, single_pointwise("add"));
        auto external_relu = add_pointwise(p1, "main:pointwise1", {d}, single_pointwise("relu"));
        auto dot2          = mm->add_instruction(migraphx::make_op("dot"), add, external_relu);
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2);
        mm->add_return({add, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto c     = mm->add_parameter("c", s3);
        auto d     = mm->add_parameter("d", s4);
        auto fused = add_mlir(
            p2,
            "main:pointwise1:mlir_main:pointwise0_geg",
            {d, a, b, c},
            [=](auto* pm, const auto& inputs) {
                auto external_relu = pm->add_instruction(migraphx::make_op("relu"), inputs[0]);
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[1], inputs[2]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[3]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, external_relu);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, add});
            });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto external_t = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, external_t});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_add_multi_user_dot_input_used_before_in_chain,
               "GEG multi-output intermediates not supported")
// GEG fusion has two outputs, E has external user
// Base case for inputs being defined within the span of will-be-fused ops, including
// longer chain of logic, for both cases of input fusion. When enabled,
// the mul gets fused into the GEG fusion.
// not currently supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto c   = mm->add_parameter("c", s3);
        auto d   = mm->add_parameter("d", s4);

        auto external_relu = add_pointwise(p1, "main:pointwise1", {d}, single_pointwise("relu"));
        auto external_mul =
            add_pointwise(p1, "main:pointwise2", {external_relu, d}, single_pointwise("mul"));

        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, c}, single_pointwise("add"));
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), add, external_mul);
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2);
        mm->add_return({add, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm           = p2.get_main_module();
        auto a             = mm->add_parameter("a", s1);
        auto b             = mm->add_parameter("b", s2);
        auto c             = mm->add_parameter("c", s3);
        auto d             = mm->add_parameter("d", s4);
        auto external_relu = add_pointwise(p2, "main:pointwise1", {d}, single_pointwise("relu"));
        auto external_mul =
            add_pointwise(p2, "main:pointwise2", {external_relu, d}, single_pointwise("mul"));
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_geg",
                     {a, b, c, external_mul},
                     [=](auto* pm, const auto& inputs) {
                         auto dot1 =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                         auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, inputs[3]);
                         return std::make_tuple(dot1->get_operator(),
                                                std::vector<migraphx::instruction_ref>{dot2, add});
                     });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto external_t = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, external_t});
    }
    migraphx::program p3;
    {
        auto* mm           = p3.get_main_module();
        auto a             = mm->add_parameter("a", s1);
        auto b             = mm->add_parameter("b", s2);
        auto c             = mm->add_parameter("c", s3);
        auto d             = mm->add_parameter("d", s4);
        auto external_relu = add_pointwise(p3, "main:pointwise1", {d}, single_pointwise("relu"));
        auto fused         = add_mlir(
            p3,
            "main:pointwise2:mlir_main:pointwise0_geg",
            {external_relu, d, a, b, c},
            [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[2], inputs[3]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[4]);
                auto mul  = pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, mul);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, add});
            });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto external_t = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, external_t});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    if(migraphx::enabled(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION{}))
        EXPECT(p1.sort() == p3.sort());
    else
        EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_add_multi_user_dot_input_used_after_in_chain,
               "GEG multi-output intermediates not supported")
// GEG fusion has two outputs, E has external user
// Testing inputs being defined within the span of will-be-fused ops, including
// longer chain of logic
// not currently supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto c   = mm->add_parameter("c", s3);
        auto d   = mm->add_parameter("d", s4);

        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, c}, single_pointwise("add"));
        auto external_relu = add_pointwise(p1, "main:pointwise1", {d}, single_pointwise("relu"));
        auto external_mul =
            add_pointwise(p1, "main:pointwise2", {external_relu, d}, single_pointwise("mul"));
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), add, external_mul);
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2);
        mm->add_return({add, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm           = p2.get_main_module();
        auto a             = mm->add_parameter("a", s1);
        auto b             = mm->add_parameter("b", s2);
        auto c             = mm->add_parameter("c", s3);
        auto d             = mm->add_parameter("d", s4);
        auto external_relu = add_pointwise(p2, "main:pointwise1", {d}, single_pointwise("relu"));
        auto external_mul =
            add_pointwise(p2, "main:pointwise2", {external_relu, d}, single_pointwise("mul"));
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_geg",
                     {a, b, c, external_mul},
                     [=](auto* pm, const auto& inputs) {
                         auto dot1 =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                         auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, inputs[3]);
                         return std::make_tuple(dot1->get_operator(),
                                                std::vector<migraphx::instruction_ref>{dot2, add});
                     });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto external_t = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, external_t});
    }
    migraphx::program p3;
    {
        auto* mm           = p3.get_main_module();
        auto a             = mm->add_parameter("a", s1);
        auto b             = mm->add_parameter("b", s2);
        auto c             = mm->add_parameter("c", s3);
        auto d             = mm->add_parameter("d", s4);
        auto external_relu = add_pointwise(p3, "main:pointwise1", {d}, single_pointwise("relu"));
        auto fused         = add_mlir(
            p3,
            "main:pointwise2:mlir_main:pointwise0_geg",
            {external_relu, d, a, b, c},
            [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[2], inputs[3]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[4]);
                auto mul  = pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, mul);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, add});
            });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto external_t = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, external_t});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    if(migraphx::enabled(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION{}))
        EXPECT(p1.sort() == p3.sort());
    else
        EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_pw_multi_user_dot, "GEG multi-output intermediates not supported")
// GEG fusion has two outputs, E has external user, E is multiple elemwise ops
// not currently supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s3);
        auto d    = mm->add_parameter("d", s3);
        auto e    = mm->add_parameter("e", s4);
        auto dot1 =
            mm->add_instruction(migraphx::make_op("dot"), a, b); // {1024, 3} x {3, 4} = {1024, 4}
        auto elemwise =
            add_pointwise(p1, "main:pointwise0", {dot1, c, d}, [=](auto* pm, const auto& inputs) {
                auto add = pm->add_instruction(migraphx::make_op("add"),
                                               inputs.at(0),
                                               inputs.at(1)); // {1024, 4} + {1024, 4} = {1024, 4}
                return pm->add_instruction(migraphx::make_op("mul"),
                                           add,
                                           inputs.at(2)); // {1024, 4} x {1024, 4} = {1024, 4}
            });
        auto dot2 = mm->add_instruction(
            migraphx::make_op("dot"), elemwise, e); // {1024, 4} * {4, 2} = {1024, 2}
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2);
        mm->add_return({elemwise, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto c     = mm->add_parameter("c", s3);
        auto d     = mm->add_parameter("d", s3);
        auto e     = mm->add_parameter("e", s4);
        auto fused = add_mlir(
            p2, "mlir_main:pointwise0_geg", {a, b, c, d, e}, [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                auto mul  = pm->add_instruction(migraphx::make_op("mul"), add, inputs[3]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), mul, inputs[4]);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, mul});
            });
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_mul =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_mul, transpose});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_multi_user_add_dot, "GEG multi-output intermediates not supported")
// GEG fusion has two outputs (first G has external user)
// not currently supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s3);
        auto d    = mm->add_parameter("d", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, c}, single_pointwise("add"));
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), add, d);
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot1);
        mm->add_return({dot2, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto c     = mm->add_parameter("c", s3);
        auto d     = mm->add_parameter("d", s4);
        auto fused = add_mlir(
            p2, "mlir_main:pointwise0_geg", {a, b, c, d}, [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, inputs[3]);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, dot1});
            });
        auto get_dot1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot1);
        mm->add_return({get_dot2, transpose});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE_SKIP(dot_add_dot_both_multi_user, "GEG multi-output intermediates not supported")
// GEG fusion has three outputs (first G has external user, E has external user)
// not currently supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s3);
        auto d    = mm->add_parameter("d", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, c}, single_pointwise("add"));
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), add, d);
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot1);
        mm->add_return({add, dot2, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto c     = mm->add_parameter("c", s3);
        auto d     = mm->add_parameter("d", s4);
        auto fused = add_mlir(
            p2, "mlir_main:pointwise0_geg", {a, b, c, d}, [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), add, inputs[3]);
                return std::make_tuple(dot1->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot2, add, dot1});
            });
        auto get_dot1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), fused);
        auto get_elemwise =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot1);
        mm->add_return({get_elemwise, get_dot2, transpose});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_add_relu_dot)
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto x    = mm->add_parameter("x", s3);
        auto y    = mm->add_parameter("y", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto pw1 =
            add_pointwise(p1, "main:pointwise0", {dot1, x}, [=](auto* pm, const auto& inputs) {
                auto add = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("relu"), add);
            });
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), pw1, y);
        mm->add_return({dot2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x   = mm->add_parameter("x", s3);
        auto y   = mm->add_parameter("y", s4);
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_mlir_dot1_geg",
                     {a, b, x, y},
                     [=](auto* pm, const auto& inputs) {
                         auto dot1 =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                         auto relu = pm->add_instruction(migraphx::make_op("relu"), add);
                         auto dot2 = pm->add_instruction(migraphx::make_op("dot"), relu, inputs[3]);
                         return std::make_tuple(dot2->get_operator(), dot2);
                     });
        mm->add_return({fused});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_dot_add)
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {4, 2}};
    migraphx::shape s4{migraphx::shape::half_type, {1024, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto y    = mm->add_parameter("y", s3);
        auto z    = mm->add_parameter("z", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), dot1, y);
        auto pw2  = add_pointwise(p1, "main:pointwise0", {dot2, z}, single_pointwise("add"));
        mm->add_return({pw2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto y   = mm->add_parameter("y", s3);
        auto z   = mm->add_parameter("z", s4);
        auto fused =
            add_mlir(p2,
                     "mlir_dot0_mlir_main:pointwise0_geg",
                     {a, b, y, z},
                     [=](auto* pm, const auto& inputs) {
                         auto dot1 =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                         auto dot2 = pm->add_instruction(migraphx::make_op("dot"), dot1, inputs[2]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), dot2, inputs[3]);
                         return std::make_tuple(add->get_operator(), add);
                     });
        mm->add_return({fused});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_dot)
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {4, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto y    = mm->add_parameter("y", s3);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), dot1, y);
        mm->add_return({dot2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto y   = mm->add_parameter("y", s3);
        auto fused =
            add_mlir(p2, "mlir_dot0_mlir_dot1_geg", {a, b, y}, [=](auto* pm, const auto& inputs) {
                auto dot1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto dot2 = pm->add_instruction(migraphx::make_op("dot"), dot1, inputs[2]);
                return std::make_tuple(dot2->get_operator(), dot2);
            });
        mm->add_return({fused});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_dot_pointwise_geg)
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {4, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {1024, 4}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1); // {1024,3}
        auto b   = mm->add_parameter("b", s2); // {3,4}
        auto c   = mm->add_parameter("c", s3); // {4,4}
        auto d   = mm->add_parameter("d", s4); // {1024,4}
        auto dot1 =
            mm->add_instruction(migraphx::make_op("dot"), a, b); // {1024,3} x {3,4} = {1024,4}
        auto dot2 =
            mm->add_instruction(migraphx::make_op("dot"), dot1, c); // {1024,4} x {4,4} = {1024,4}
        auto add = add_pointwise(p1, "main:pointwise0", {dot2, d}, single_pointwise("add"));
        mm->add_return({add}); // {1024,4}
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto c   = mm->add_parameter("c", s3);
        auto d   = mm->add_parameter("d", s4);
        auto fused =
            add_mlir(p2,
                     "mlir_dot0_mlir_main:pointwise0_geg",
                     {a, b, c, d},
                     [=](auto* pm, const auto& inputs) {
                         auto dot1 =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                         auto dot2 = pm->add_instruction(migraphx::make_op("dot"), dot1, inputs[2]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), dot2, inputs[3]);
                         return std::make_tuple(add->get_operator(), add);
                     });
        mm->add_return({fused});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_add_relu_dot_add_relu)
// criteoterabyte use case
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {1024, 4}};
    migraphx::shape s4{migraphx::shape::half_type, {4, 2}};
    migraphx::shape s5{migraphx::shape::half_type, {1024, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto x    = mm->add_parameter("x", s3);
        auto y    = mm->add_parameter("y", s4);
        auto z    = mm->add_parameter("z", s5);
        auto dot1 = mm->add_instruction(
            migraphx::make_op("dot"), a, b); // {1024,4}, m + 1000 > avg (n, k, gemmO)
        auto pw1 =
            add_pointwise(p1, "main:pointwise0", {dot1, x}, [=](auto* pm, const auto& inputs) {
                auto add = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("relu"), add);
            });                                                            // {1024,4}
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), pw1, y); // {1024,2}
        auto pw2 =
            add_pointwise(p1, "main:pointwise1", {dot2, z}, [=](auto* pm, const auto& inputs) {
                auto add = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("relu"), add);
            }); // {1024,2}
        mm->add_return({pw2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s1);
        auto b     = mm->add_parameter("b", s2);
        auto x     = mm->add_parameter("x", s3);
        auto y     = mm->add_parameter("y", s4);
        auto z     = mm->add_parameter("z", s5);
        auto fused = add_mlir(
            p2,
            "mlir_main:pointwise0_mlir_main:pointwise1_geg",
            {a, b, x, y, z},
            [=](auto* pm, const auto& inputs) {
                auto dot1  = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add1  = pm->add_instruction(migraphx::make_op("add"), dot1, inputs[2]);
                auto relu1 = pm->add_instruction(migraphx::make_op("relu"), add1);
                auto dot2  = pm->add_instruction(migraphx::make_op("dot"), relu1, inputs[3]);
                auto add2  = pm->add_instruction(migraphx::make_op("add"), dot2, inputs[4]);
                auto relu2 = pm->add_instruction(migraphx::make_op("relu"), add2);
                return std::make_tuple(relu2->get_operator(), relu2);
            });
        mm->add_return({fused});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_dot_add_with_gemm_multi_out)
// Second submodule (dot+add) has multi-out where the inner dot is also returned
// This scenario *is* supported in rocMLIR
{
    migraphx::shape s1{migraphx::shape::half_type, {1024, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {3, 4}};
    migraphx::shape s3{migraphx::shape::half_type, {4, 2}};
    migraphx::shape s4{migraphx::shape::half_type, {1024, 2}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto y    = mm->add_parameter("y", s3);
        auto z    = mm->add_parameter("z", s4);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);    // {1024,4}
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), dot1, y); // {1024,2}
        auto add =
            add_pointwise(p1, "main:pointwise0", {dot2, z}, single_pointwise("add")); // {1024,2}
        // dot2 has another user (creating multi-out scenario)
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dot2); // {2,1024}
        mm->add_return({add, transpose});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto y   = mm->add_parameter("y", s3);
        auto z   = mm->add_parameter("z", s4);
        auto fused =
            add_mlir(p2,
                     "mlir_dot0_mlir_main:pointwise0_geg",
                     {a, b, y, z},
                     [=](auto* pm, const auto& inputs) {
                         auto dot1 =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                         auto dot2 = pm->add_instruction(migraphx::make_op("dot"), dot1, inputs[2]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), dot2, inputs[3]);
                         return std::make_tuple(add->get_operator(),
                                                std::vector<migraphx::instruction_ref>{add, dot2});
                     });
        auto get_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto get_dot2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), get_dot2);
        mm->add_return({get_add, transpose});
    }
    if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dyn_dot)
{
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4}, {6, 6}}};
    migraphx::shape s2{migraphx::shape::float_type, {6, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto a      = mm->add_parameter("a", s1);
        auto b      = mm->add_parameter("b", s2);
        auto a_cont = mm->add_instruction(migraphx::make_op("contiguous"), a);

        auto fused =
            add_mlir(p2, "mlir_dot0", {a_cont, b}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dyn_conv)
{
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4}, {56, 56}, {8, 64}, {8, 64}}};
    migraphx::shape s2{migraphx::shape::float_type, {14, 56, 3, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s1);
        auto w    = mm->add_parameter("w", s2);
        auto conv = mm->add_instruction(migraphx::make_op("convolution"), x, w);
        mm->add_return({conv});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto x      = mm->add_parameter("x", s1);
        auto w      = mm->add_parameter("w", s2);
        auto x_cont = mm->add_instruction(migraphx::make_op("contiguous"), x);
        auto conv   = add_mlir(
            p2, "mlir_convolution0", {x_cont, w}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto c =
                    pm->add_instruction(migraphx::make_op("convolution"), inputs[0], inputs[1]);
                return std::make_tuple(c->get_operator(), c);
            });
        mm->add_return({conv});
    }
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[])
{
    if(migraphx::gpu::mlir_enabled())
    {
        test::run(argc, argv);
    }
    return 0;
}
