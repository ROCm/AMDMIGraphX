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

#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/stringutils.hpp>

#include <test.hpp>
#include <pointwise.hpp>

// Two adds fused into a single pointwise op via fuse_pointwise.
// Both symbols should appear on the fused pointwise instruction.
//
//  Before:                          After:
//
//   x   y                           x  y  z
//    \ /                              \ | /
//    add  {add1}                   pointwise  {add1, add2}
//     |  z                             |
//     | /                           @return
//    add  {add2}
//     |
//   @return
//
TEST_CASE(pw_double_add)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm                       = p1.get_main_module();
        auto x                         = mm->add_parameter("x", s);
        auto y                         = mm->add_parameter("y", s);
        auto z                         = mm->add_parameter("z", s);
        migraphx::instruction_ref add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_debug_symbols(add1, {"add1"});
        migraphx::instruction_ref add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        mm->add_debug_symbols(add2, {"add2"});
        mm->add_return({add2});
    }
    migraphx::run_passes(p1, {migraphx::fuse_pointwise{}, migraphx::dead_code_elimination{}});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x, y, z}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        mm->add_debug_symbols(fadd, {"add1", "add2"});
        mm->add_return({fadd});
    }
    EXPECT(p1 == p2);
}

//  Before:                       After:
//
//    x   y                        x   y
//     \ /                          \ /
//     add1 {add1}               pointwise  {add1, add2, add3, add4}
//    / \                            |
//   x   y                        @return
//   |   |
//  add2 add3 {add2} {add3}
//    \  /
//    add4 {add4}
//      |
//   @return
//
TEST_CASE(pw_used_twice_fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_debug_symbols(add1, {"onnx:add1"});
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, x);
        mm->add_debug_symbols(add2, {"onnx:add2"});
        auto add3 = mm->add_instruction(migraphx::make_op("add"), add1, y);
        mm->add_debug_symbols(add3, {"onnx:add3"});
        auto add4 = mm->add_instruction(migraphx::make_op("add"), add2, add3);
        mm->add_debug_symbols(add4, {"onnx:add4"});
        mm->add_return({add4});
    }
    migraphx::run_passes(p1, {migraphx::fuse_pointwise{}, migraphx::dead_code_elimination{}});

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto fadd = add_pointwise(p2, "main:pointwise0", {x, y}, [=](auto* pm, const auto& inputs) {
            auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
            auto add2 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[0]);
            auto add3 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[1]);
            return pm->add_instruction(migraphx::make_op("add"), add2, add3);
        });
        mm->add_debug_symbols(fadd, {"onnx:add1", "onnx:add2", "onnx:add3", "onnx:add4"});
        mm->add_return({fadd});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Debug symbols should not propagate above the fusion boundary.
// The gemm (dot) keeps its own symbol; only the fused adds merge.
//
//  Before:                          After:
//
//   x   a                            x    a
//    \ /                              \ /
//    dot  {gemm1}                    dot  {gemm1}
//     |  y                          /
//     | /                      gemm  y    z
//    add  {add1}                  \  |  /
//     |  z                      pointwise  {add1, add2}
//     | /                            |
//    add  {add2}                  @return
//     |
//   @return
//
TEST_CASE(gemm_add_add)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s1);
        auto y    = mm->add_parameter("y", s1);
        auto z    = mm->add_parameter("z", s1);
        auto a    = mm->add_literal(migraphx::generate_literal(s2, 0));
        auto gemm = mm->add_instruction(migraphx::make_op("dot"), x, a);
        mm->add_debug_symbols(gemm, {"gemm1"});
        auto add1 = mm->add_instruction(migraphx::make_op("add"), gemm, y);
        mm->add_debug_symbols(add1, {"add1"});
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        mm->add_debug_symbols(add2, {"add2"});
        mm->add_return({add2});
    }
    migraphx::run_passes(p1, {migraphx::fuse_pointwise{}, migraphx::dead_code_elimination{}});

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s1);
        auto y    = mm->add_parameter("y", s1);
        auto z    = mm->add_parameter("z", s1);
        auto a    = mm->add_literal(migraphx::generate_literal(s2, 0));
        auto gemm = mm->add_instruction(migraphx::make_op("dot"), x, a);
        mm->add_debug_symbols(gemm, {"gemm1"});
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {gemm, y, z}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        mm->add_debug_symbols(fadd, {"add1", "add2"});
        mm->add_return({fadd});
    }
    EXPECT(p1 == p2);
}

// Horizontal fusion of two dot ops sharing the same input via
// simplify_algebra. The two dots are fused into concat + single dot + slices.
// Each new instruction inherits the symbols of the original dots it derives
// from (e.g. the concat and fused dot carry both "gemm1" and "gemm2").
//
//  Before:                          After:
//
//     a     input     b              a       b
//     \   /      \   /                \     /
//     dot {g1}   dot {g2}              concat    {g1, g2}
//       \       /                 input  |
//        \     /                     \   |
//        add {sum}                    dot      {g1, g2}
//                                   /    \
//                         slice{g1, g2}  slice{g1, g2}
//                                  \       /
//                                   add {sum}
//
TEST_CASE(horiz_fusion_dot)
{
    auto type = migraphx::shape::int32_type;
    auto s    = migraphx::shape{type, {3, 2, 2}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(s, 0));
        auto b     = m1.add_literal(migraphx::generate_literal(s, 1));
        auto x     = m1.add_instruction(migraphx::make_op("dot"), input, a);
        m1.add_debug_symbols(x, {"gemm1"});
        auto y = m1.add_instruction(migraphx::make_op("dot"), input, b);
        m1.add_debug_symbols(y, {"gemm2"});
        auto sum = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_debug_symbols(sum, {"sum"});
        m1.add_return({sum});
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto input  = m2.add_parameter("input", s);
        auto a      = m2.add_literal(migraphx::generate_literal(s, 0));
        auto b      = m2.add_literal(migraphx::generate_literal(s, 1));
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), a, b);
        m2.add_debug_symbols(concat, {"gemm1", "gemm2"});
        auto dot = m2.add_instruction(migraphx::make_op("dot"), input, concat);
        m2.add_debug_symbols(dot, {"gemm1", "gemm2"});
        auto x = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), dot);
        m2.add_debug_symbols(x, {"gemm1", "gemm2"});
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {4}}}), dot);
        m2.add_debug_symbols(y, {"gemm1", "gemm2"});
        auto sum = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_debug_symbols(sum, {"sum"});
        m2.add_return({sum});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Tests symbol propagation through add reassociation in simplify_algebra
// (find_double_add_lit_broadcast). Checks add(add(x,1), add(y,2)) -> (add(add(x,y), add(1,2)).
//
//  Before:                          After:
//
//   x   1    y   2                    1   2       x   y
//    \ /      \ /                      \ /         \ /
//   add1{a1} add2{a2}                 add{a0,a1,a2} add{a0,a1,a2}
//      \     /                              \       /
//      add0{a0}                             add{a0,a1,a2}
//
TEST_CASE(simplify_add_debug_symbols)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), x, one);
        m1.add_debug_symbols(sum1, {"onnx:add1"});
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), y, two);
        m1.add_debug_symbols(sum2, {"onnx:add2"});
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        m1.add_debug_symbols(sum3, {"onnx:add0"});
        m1.add_return({sum3});
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        m2.add_debug_symbols(sum1, {"onnx:add0", "onnx:add1", "onnx:add2"});
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_debug_symbols(sum2, {"onnx:add0", "onnx:add1", "onnx:add2"});
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum2, sum1);
        m2.add_debug_symbols(sum3, {"onnx:add0", "onnx:add1", "onnx:add2"});
        m2.add_return({sum3});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Tests the replace_instruction(ins, rep) overload via find_unit_ops which
// simplifies add(relu(x), broadcast(0)) to relu(x).
//
//  Before:                       After:
//
//    x     0                       x
//    |     |                       |
//  relu   bcast                  relu  {add, relu}
//  {relu} (0.0)
//     \   /
//     add  {add}
//
TEST_CASE(replace_with_insref_debug_symbols)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto zero = m1.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        auto bcast =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), zero);
        auto relu_x = m1.add_instruction(migraphx::make_op("relu"), x);
        m1.add_debug_symbols(relu_x, {"onnx:relu"});
        auto add_r = m1.add_instruction(migraphx::make_op("add"), relu_x, bcast);
        m1.add_debug_symbols(add_r, {"onnx:add"});
        m1.add_return({add_r});
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto relu_x = m2.add_instruction(migraphx::make_op("relu"), x);
        m2.add_debug_symbols(relu_x, {"onnx:add", "onnx:relu"});
        m2.add_return({relu_x});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Tests that debug_symbols propagate through the dead-code chain.
//
//  Before:                       After:
//
//    x                             x   y
//    |                              \ /
//   relu  {relu}                   mul  {add, relu}
//    |  y
//    | /
//   add   {add}
//    |          (relu becomes dead code, removed by DCE)
//   pass
//
TEST_CASE(gather_replace_chain_debug_symbols)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto relu_x = m1.add_instruction(migraphx::make_op("relu"), x);
        m1.add_debug_symbols(relu_x, {"onnx:relu"});
        auto add_r = m1.add_instruction(migraphx::make_op("add"), relu_x, y);
        m1.add_debug_symbols(add_r, {"onnx:add"});
        m1.add_return({add_r});

        auto mul_r = m1.insert_instruction(add_r, migraphx::make_op("mul"), x, y);
        m1.replace_instruction(add_r, mul_r);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", s);
        auto y     = m2.add_parameter("y", s);
        auto mul_r = m2.add_instruction(migraphx::make_op("mul"), x, y);
        m2.add_debug_symbols(mul_r, {"onnx:add", "onnx:relu"});
        m2.add_return({mul_r});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Tests the distributive law transform in simplify_algebra (find_mul_add):
// mul(add(3, x), 2) -> add(mul(2, x), mul(2, 3)).
//
//  Before:                           After:
//
//    3   x                           2   x        2   3
//     \ /                             \ /          \ /
//    add  {add}                      mul{add,mul}  mul{add,mul}
//     |  2                                 \       /
//     | /                                  add{add,mul}
//    mul  {mul}
//
TEST_CASE(simplify_mul_add_debug_symbols)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one = m1.add_literal(3);
        auto two = m1.add_literal(2);
        auto sum = m1.add_instruction(migraphx::make_op("add"), one, x);
        m1.add_debug_symbols(sum, {"onnx:add"});
        auto mul = m1.add_instruction(migraphx::make_op("mul"), sum, two);
        m1.add_debug_symbols(mul, {"onnx:mul"});
        m1.add_return({mul});
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(3);
        auto two  = m2.add_literal(2);
        auto mul1 = m2.add_instruction(migraphx::make_op("mul"), two, x);
        m2.add_debug_symbols(mul1, {"onnx:add", "onnx:mul"});
        auto mul2 = m2.add_instruction(migraphx::make_op("mul"), two, one);
        m2.add_debug_symbols(mul2, {"onnx:add", "onnx:mul"});
        auto sum = m2.add_instruction(migraphx::make_op("add"), mul1, mul2);
        m2.add_debug_symbols(sum, {"onnx:add", "onnx:mul"});
        m2.add_return({sum});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Tests symbol propagation through find_div_const in simplify_algebra:
// div(x, c) -> mul(x, recip(c)).
//
//  Before:                       After:
//
//   x   c                         c
//    \ /                          |
//   div  {div}                  recip  {div}
//                                x  |
//                                 \ |
//                                 mul  {div}
//
TEST_CASE(simplify_div_const_debug_symbols)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto c =
            m1.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2, 3}},
                                             {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}});
        auto div_r = m1.add_instruction(migraphx::make_op("div"), x, c);
        m1.add_debug_symbols(div_r, {"onnx:div"});
        m1.add_return({div_r});
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        auto c =
            m2.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2, 3}},
                                             {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}});
        auto recip = m2.add_instruction(migraphx::make_op("recip"), c);
        m2.add_debug_symbols(recip, {"onnx:div"});
        auto mul_r = m2.add_instruction(migraphx::make_op("mul"), x, recip);
        m2.add_debug_symbols(mul_r, {"onnx:div"});
        m2.add_return({mul_r});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Verifies that debug symbols appear in the module's printed/serialized
// output using the expected comment format produced by instruction::print.
//
//  Printed output includes:
//    @2 = add(@0,@1) -> float_type, {2, 3} # sym_a, sym_b #
//
TEST_CASE(debug_symbols_in_print)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x   = m.add_parameter("x", s);
    auto y   = m.add_parameter("y", s);
    auto add = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_debug_symbols(add, {"sym_a", "sym_b"});
    m.add_return({add});

    auto str = migraphx::to_string(m);
    EXPECT(str.find("# sym_a, sym_b #") != std::string::npos);
}

// -----------------------------------------------------------------------
// Direct replace_instruction / batch_replace_instruction tests
// -----------------------------------------------------------------------

// replace_instruction(ins, op, args) -- simple in-place, no splice chain.
// The replaced instruction retains its own debug symbols.
//
//  Before:                       After:
//
//    x   y                        x   y
//     \ /                          \ /
//    add  {sym_add}               mul  {sym_add}
//     |                            |
//   @return                      @return
//
TEST_CASE(replace_op_args_no_splice)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x   = m.add_parameter("x", s);
    auto y   = m.add_parameter("y", s);
    auto add = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_debug_symbols(add, {"sym_add"});
    m.add_return({add});

    m.replace_instruction(add, migraphx::make_op("mul"), x, y);

    EXPECT(add->name() == "mul");
    EXPECT(add->get_debug_symbols() == std::set<std::string>{"sym_add"});
}

// replace_instruction(ins, op, args) -- splice chain propagation.
// relu only outputs to neg, so relu is in neg's splice chain.  When neg
// is replaced with abs(x), relu's symbols propagate to the replacement.
//
//  Before:                       After:
//
//    x                             x
//    |                             |
//   relu  {relu_sym}             abs  {neg_sym, relu_sym}
//    |                             |
//   neg   {neg_sym}             @return
//    |
//  @return
//
TEST_CASE(replace_op_args_spliced_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x      = m.add_parameter("x", s);
    auto relu_x = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_debug_symbols(relu_x, {"relu_sym"});
    auto neg_r = m.add_instruction(migraphx::make_op("neg"), relu_x);
    m.add_debug_symbols(neg_r, {"neg_sym"});
    m.add_return({neg_r});

    m.replace_instruction(neg_r, migraphx::make_op("abs"), x);

    EXPECT(neg_r->name() == "abs");
    std::set<std::string> expected{"neg_sym", "relu_sym"};
    EXPECT(neg_r->get_debug_symbols() == expected);
}

// replace_instruction(ins, op, args, module_args) -- in-place replace with
// module arguments.  The debug symbol logic is identical to the args-only
// overload; this test exercises the separate code path.
//
//  Before:                       After:
//
//    cond                          cond
//     |                             |
//    if  {sym_if}                  if  {sym_if}    (different sub-modules)
//     |                             |
//   get[0]                        get[0]
//     |                             |
//   @return                       @return
//
TEST_CASE(replace_op_args_module_args)
{
    migraphx::shape cond_s{migraphx::shape::bool_type};
    migraphx::shape s{migraphx::shape::float_type, {5}};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto cond = mm->add_parameter("cond", cond_s);

    auto* then_mod1       = p.create_module("then1");
    std::vector<float> d1 = {1, 2, 3, 4, 5};
    auto l1               = then_mod1->add_literal(migraphx::literal(s, d1));
    then_mod1->add_return({l1});

    auto* else_mod1       = p.create_module("else1");
    std::vector<float> d2 = {5, 4, 3, 2, 1};
    auto l2               = else_mod1->add_literal(migraphx::literal(s, d2));
    else_mod1->add_return({l2});

    auto if_ins = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod1, else_mod1});
    mm->add_debug_symbols(if_ins, {"sym_if"});
    auto r = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), if_ins);
    mm->add_return({r});

    auto* then_mod2       = p.create_module("then2");
    std::vector<float> d3 = {10, 20, 30, 40, 50};
    auto l3               = then_mod2->add_literal(migraphx::literal(s, d3));
    then_mod2->add_return({l3});

    auto* else_mod2       = p.create_module("else2");
    std::vector<float> d4 = {50, 40, 30, 20, 10};
    auto l4               = else_mod2->add_literal(migraphx::literal(s, d4));
    else_mod2->add_return({l4});

    mm->replace_instruction(if_ins,
                            migraphx::make_op("if"),
                            std::vector<migraphx::instruction_ref>{cond},
                            std::vector<migraphx::module_ref>{then_mod2, else_mod2});

    EXPECT(if_ins->get_debug_symbols() == std::set<std::string>{"sym_if"});
}

// replace_instruction(ins, rep) -- redirect outputs from ins to rep.
// rep inherits the debug symbols of ins.
//
//  Before:                       After (add becomes dead):
//
//    x   y                        x   y
//    |\ /|                         \ /
//    | X |                         mul  {sym_add}
//    |/ \|                          |
//   mul  add  {sym_add}          @return
//         |
//       @return
//
TEST_CASE(replace_with_insref_no_splice)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x   = m.add_parameter("x", s);
    auto y   = m.add_parameter("y", s);
    auto mul = m.add_instruction(migraphx::make_op("mul"), x, y);
    auto add = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_debug_symbols(add, {"sym_add"});
    m.add_return({add});

    m.replace_instruction(add, mul);

    EXPECT(mul->get_debug_symbols() == std::set<std::string>{"sym_add"});
}

// batch_replace_instruction -- single element with splice chain.
// Same topology as replace_op_args_spliced_inputs but via the batch API.
//
//  Before:                       After:
//
//    x                             x
//    |                             |
//   relu  {relu_sym}             abs  {neg_sym, relu_sym}
//    |                             |
//   neg   {neg_sym}             @return
//    |
//  @return
//
TEST_CASE(batch_replace_single_with_splice)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x      = m.add_parameter("x", s);
    auto relu_x = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_debug_symbols(relu_x, {"relu_sym"});
    auto neg_r = m.add_instruction(migraphx::make_op("neg"), relu_x);
    m.add_debug_symbols(neg_r, {"neg_sym"});
    m.add_return({neg_r});

    auto results = m.batch_replace_instruction({{neg_r, migraphx::make_op("abs"), {x}, {}}});

    EXPECT(results.size() == 1);
    std::set<std::string> expected{"neg_sym", "relu_sym"};
    EXPECT(results[0]->get_debug_symbols() == expected);
}

// batch_replace_instruction -- two simultaneous replacements.  The batch
// collects symbols from ALL replaced instructions and propagates the
// merged set to every new-splice instruction.  With individual replaces
// mul would only get {add1_sym} and div only {add2_sym}; the batch
// merges them so both receive the combined set.
//
//  Before:                           After:
//
//    x   y                           x   y
//    |\ /|                           |\ /|
//    | X |                           | X |
//    |/ \|                           |/ \|
//  add1   add2                     mul    div   {add1_sym, add2_sym}
//  {add1_sym} {add2_sym}            \     /
//      \     /                      @return
//      @return
//
TEST_CASE(batch_replace_multi_merges_symbols)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x    = m.add_parameter("x", s);
    auto y    = m.add_parameter("y", s);
    auto add1 = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_debug_symbols(add1, {"add1_sym"});
    auto add2 = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_debug_symbols(add2, {"add2_sym"});
    m.add_return({add1, add2});

    auto results = m.batch_replace_instruction({
        {add1, migraphx::make_op("mul"), {x, y}, {}},
        {add2, migraphx::make_op("div"), {x, y}, {}},
    });

    EXPECT(results.size() == 2);
    std::set<std::string> expected{"add1_sym", "add2_sym"};
    EXPECT(results[0]->get_debug_symbols() == expected);
    EXPECT(results[1]->get_debug_symbols() == expected);
}

// -----------------------------------------------------------------------
// module::remove_debug_symbols tests
// -----------------------------------------------------------------------

TEST_CASE(remove_single_symbol)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x   = m.add_parameter("x", s);
    auto y   = m.add_parameter("y", s);
    auto add = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_debug_symbols(add, {"sym_a", "sym_b"});
    m.add_return({add});

    EXPECT(m.has_debug_symbols());
    m.remove_debug_symbols(add);

    EXPECT(add->get_debug_symbols().empty());
    EXPECT(not m.has_debug_symbols());
}

TEST_CASE(remove_noop_no_symbols)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x   = m.add_parameter("x", s);
    auto y   = m.add_parameter("y", s);
    auto add = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_return({add});

    EXPECT(not m.has_debug_symbols());
    m.remove_debug_symbols(add);

    EXPECT(add->get_debug_symbols().empty());
    EXPECT(not m.has_debug_symbols());
}

TEST_CASE(remove_one_of_two)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x    = m.add_parameter("x", s);
    auto y    = m.add_parameter("y", s);
    auto add1 = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_debug_symbols(add1, {"sym_add1"});
    auto add2 = m.add_instruction(migraphx::make_op("add"), add1, y);
    m.add_debug_symbols(add2, {"sym_add2"});
    m.add_return({add2});

    EXPECT(m.has_debug_symbols());
    m.remove_debug_symbols(add1);

    EXPECT(add1->get_debug_symbols().empty());
    EXPECT(add2->get_debug_symbols() == std::set<std::string>{"sym_add2"});
    EXPECT(m.has_debug_symbols());
}

TEST_CASE(remove_then_re_add)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x   = m.add_parameter("x", s);
    auto y   = m.add_parameter("y", s);
    auto add = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_debug_symbols(add, {"old_sym"});
    m.add_return({add});

    m.remove_debug_symbols(add);
    EXPECT(add->get_debug_symbols().empty());
    EXPECT(not m.has_debug_symbols());

    m.add_debug_symbols(add, {"new_sym"});
    EXPECT(add->get_debug_symbols() == std::set<std::string>{"new_sym"});
    EXPECT(m.has_debug_symbols());
}

TEST_CASE(remove_idempotent)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    auto x   = m.add_parameter("x", s);
    auto y   = m.add_parameter("y", s);
    auto add = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_debug_symbols(add, {"sym"});
    m.add_return({add});

    m.remove_debug_symbols(add);
    m.remove_debug_symbols(add);

    EXPECT(add->get_debug_symbols().empty());
    EXPECT(not m.has_debug_symbols());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
