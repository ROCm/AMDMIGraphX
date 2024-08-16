/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include <pointwise.hpp>
#include <reduce.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION);

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::gpu::fuse_mlir{.enable_extra = true}, migraphx::dead_code_elimination{}});
}

template <class F>
migraphx::instruction_ref add_mlir(migraphx::program& p,
                                   const std::string& name,
                                   std::vector<migraphx::instruction_ref> inputs,
                                   std::vector<std::string> arg_names,
                                   F f)
{
    assert(inputs.size() == arg_names.size() && "One interior parameter name given per input.");
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
    pm->add_return({r});
    return mm->add_instruction(
        migraphx::make_op("gpu::mlir_op", {{"op", migraphx::to_value(root)}}), inputs, {pm});
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
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s);
        auto b     = mm->add_parameter("b", s);
        auto x     = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 3}});
        auto fused = add_mlir(
            p2,
            "mlir_main:pointwise0",
            {x, a, b},
            {"x2", "y0", "y1"},
            [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[1], inputs[2]);
                auto dot_trans = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
                auto dot_rsp = pm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 3}}}),
                                                   dot_trans);
                auto add     = pm->add_instruction(migraphx::make_op("add"), dot_rsp, inputs[0]);
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
            add_mlir(p2,
                     "mlir_main:pointwise0",
                     {x, a, b},
                     {"x2", "y0", "y1"},
                     [=](auto* pm, const auto& inputs) {
                         auto dot =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[1], inputs[2]);
                         auto add = pm->add_instruction(migraphx::make_op("add"), dot, inputs[0]);
                         return std::make_tuple(dot->get_operator(), add);
                     });
        mm->add_return({fused});
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
        auto fused = add_mlir(
            p2,
            "mlir_main:pointwise0",
            {x, a, b},
            {"x2", "y0", "y1"},
            [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[1], inputs[2]);
                auto dot_trans = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), dot);

                auto add = pm->add_instruction(migraphx::make_op("add"), dot_trans, inputs[0]);
                return std::make_tuple(dot->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot, add});
            });
        auto fused_dot_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {0, 0, 0, 1}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                fused_dot_add);
        auto dot = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
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
        auto fused = add_mlir(
            p2,
            "mlir_main:pointwise0",
            {x, a, b},
            {"x2", "y0", "y1"},
            [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[1], inputs[2]);
                auto dot_trans = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
                auto dot_unsq = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {1, 1, 5, 4}}}), dot_trans);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot_unsq, inputs[0]);
                return std::make_tuple(dot->get_operator(),
                                       std::vector<migraphx::instruction_ref>{dot, add});
            });
        auto fused_dot_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {1, 0, 0, 0}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                fused_dot_add);
        auto dot = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
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
            add_mlir(p2, "mlir_dot4", {a, b}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        auto dot2 =
            add_mlir(p2, "mlir_dot5", {dot1, c}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        auto add = add_pointwise(p2, "main:pointwise0", {dot1, dot2}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(dot_dot_pointwise_pointwise)
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
            add_mlir(p2, "mlir_dot6", {a, b}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0",
                     {x, dot1, c},
                     {"x2", "y0", "y1"},
                     [=](auto* pm, const auto& inputs) {
                         auto dot =
                             pm->add_instruction(migraphx::make_op("dot"), inputs[1], inputs[2]);
                         auto add = pm->add_instruction(migraphx::make_op("add"), dot, inputs[0]);
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
                     "main:pointwise0:mlir_dot8",
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
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s_a);
        auto b     = mm->add_parameter("b", s_b);
        auto fused = add_mlir(
            p2, "mlir_main:pointwise0", {a, b}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
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
        auto mb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}), b);
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_main:split_reduce0",
                     {mb, x, w},
                     {"x2", "y0", "y1"},
                     [=](auto* pm, const auto& inputs) {
                         auto conv = pm->add_instruction(
                             migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                             inputs[1],
                             inputs[2]);
                         auto reshape = pm->add_instruction(
                             migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
                         auto add =
                             pm->add_instruction(migraphx::make_op("add"), reshape, inputs[0]);
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
        auto mb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}), b);
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_main:split_reduce0",
                     {mb, x, w},
                     {"x2", "y0", "y1"},
                     [=](auto* pm, const auto& inputs) {
                         auto conv = pm->add_instruction(
                             migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                             inputs[1],
                             inputs[2]);
                         auto reshape = pm->add_instruction(
                             migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
                         auto add =
                             pm->add_instruction(migraphx::make_op("add"), reshape, inputs[0]);
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
        auto mb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}), b);
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_main:split_reduce0",
                     {mb, x, w1},
                     {"x2", "y0", "y1"},
                     [=](auto* pm, const auto& inputs) {
                         auto conv = pm->add_instruction(
                             migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                             inputs[1],
                             inputs[2]);
                         auto reshape = pm->add_instruction(
                             migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
                         auto add =
                             pm->add_instruction(migraphx::make_op("add"), reshape, inputs[0]);
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
        auto mean_rsp = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), mean_mb);
        auto var_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", cba->get_shape().lens()}}), var);
        auto var_rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), var_mb);
        auto cba_rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), cba);
        auto input_fused_conv = add_mlir(
            p2,
            "main:pointwise2:mlir_convolution3",
            {cba_rsp, mean_rsp, var_rsp, w2},
            {"x0", "x1", "x2", "x3"},
            [=](auto* pm, const auto& inputs) {
                auto sub =
                    pm->add_instruction(migraphx::make_op("sub"), inputs.at(0), inputs.at(1));
                auto div  = pm->add_instruction(migraphx::make_op("div"), sub, inputs.at(2));
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

int main(int argc, const char* argv[])
{
    if(migraphx::gpu::mlir_enabled())
    {
        test::run(argc, argv);
    }
    return 0;
}
