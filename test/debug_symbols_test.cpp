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
//    add  {add0}                   pointwise  {add0, add1}
//     |  z                             |
//     | /                           @return
//    add  {add1}
//     |
//   @return
//
TEST_CASE(pw_double_add)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        mm->set_use_debug_symbols();
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", s);
        auto z = mm->add_parameter("z", s);
        migraphx::instruction_ref add1;
        {
            migraphx::scoped_debug_symbols guard0(*mm, {"onnx:add0"});
            add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        }
        migraphx::instruction_ref add2;
        {
            migraphx::scoped_debug_symbols guard1(*mm, {"onnx:add1"});
            add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        }
        mm->add_return({add2});
    }
    migraphx::run_passes(p1, {migraphx::fuse_pointwise{}, migraphx::dead_code_elimination{}});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        mm->set_use_debug_symbols();
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", s);
        auto z = mm->add_parameter("z", s);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x, y, z}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        fadd->add_debug_symbols({"onnx:add0", "onnx:add1"});
        mm->add_return({fadd});
    }
    // BUG straight equality is not working even though both call migraphx::to_string
    // EXPECT(p1 == p2);
    EXPECT(to_string(p1) == to_string(p2));
}

// Diamond pattern: add1 feeds into both add2 and add3, which then feed
// into add4. All four are fused into one pointwise op. Verifies that
// symbols from every instruction in the diamond appear on the fused result.
//
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
        auto* mm = p1.get_main_module();
        mm->set_use_debug_symbols();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        add1->add_debug_symbols({"onnx:add1"});
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, x);
        add2->add_debug_symbols({"onnx:add2"});
        auto add3 = mm->add_instruction(migraphx::make_op("add"), add1, y);
        add3->add_debug_symbols({"onnx:add3"});
        auto add4 = mm->add_instruction(migraphx::make_op("add"), add2, add3);
        add4->add_debug_symbols({"onnx:add4"});
        mm->add_return({add4});
    }
    migraphx::run_passes(p1, {migraphx::fuse_pointwise{}, migraphx::dead_code_elimination{}});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        mm->set_use_debug_symbols();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto fadd = add_pointwise(p2, "main:pointwise0", {x, y}, [=](auto* pm, const auto& inputs) {
            auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
            auto add2 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[0]);
            auto add3 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[1]);
            return pm->add_instruction(migraphx::make_op("add"), add2, add3);
        });
        fadd->add_debug_symbols({"onnx:add1", "onnx:add2", "onnx:add3", "onnx:add4"});
        mm->add_return({fadd});
    }
    // BUG straight equality is not working even though both call migraphx::to_string
    EXPECT(to_string(p1.sort()) == to_string(p2.sort()));
}

// Horizontal fusion of two dot ops sharing the same input via
// simplify_algebra. The two dots are fused into concat + single dot + slices.
// Each new instruction inherits the symbols of the original dots it derives
// from (e.g. the concat and fused dot carry both "gemm1" and "gemm2").
//
//  Before:                          After:
//
//   input  a   input  b              a       b
//     \   /      \   /                \     /
//     dot {g1}   dot {g2}           concat    {g1, g2}
//       \       /                  input |
//        \     /                     \   |
//        add {sum}                    dot      {g1, g2}
//          |                         /   \
//        pass                  slice{g1}  slice{g2}
//                                  \       /
//                                   add   {sum}
//                                    |
//                                  pass
//
TEST_CASE(horiz_fusion_dot)
{
    auto type = migraphx::shape::int32_type;
    auto s    = migraphx::shape{type, {3, 2, 2}};
    migraphx::module m1;
    {
        m1.set_use_debug_symbols();
        auto input = m1.add_parameter("input", s);
        auto a     = m1.add_literal(migraphx::generate_literal(s, 0));
        auto b     = m1.add_literal(migraphx::generate_literal(s, 1));
        auto x     = m1.add_instruction(migraphx::make_op("dot"), input, a);
        x->add_debug_symbols({"gemm1"});
        auto y = m1.add_instruction(migraphx::make_op("dot"), input, b);
        y->add_debug_symbols({"gemm2"});
        auto sum = m1.add_instruction(migraphx::make_op("add"), x, y);
        sum->add_debug_symbols({"sum"});
        m1.add_instruction(pass_op{}, sum);
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        m2.set_use_debug_symbols();
        auto input  = m2.add_parameter("input", s);
        auto a      = m2.add_literal(migraphx::generate_literal(s, 0));
        auto b      = m2.add_literal(migraphx::generate_literal(s, 1));
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), a, b);
        concat->add_debug_symbols({"gemm1", "gemm2"});
        auto dot = m2.add_instruction(migraphx::make_op("dot"), input, concat);
        dot->add_debug_symbols({"gemm1", "gemm2"});
        auto x = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), dot);
        x->add_debug_symbols({"gemm1"});
        auto y = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {4}}}), dot);
        y->add_debug_symbols({"gemm2"});
        auto sum = m2.add_instruction(migraphx::make_op("add"), x, y);
        sum->add_debug_symbols({"sum"});
        m2.add_instruction(pass_op{}, sum);
    }
    // BUG straight equality is not working even though both call migraphx::to_string
    EXPECT(to_string(m1.sort()) == to_string(m2.sort()));
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
//        |                                    |
//       pass                                 pass
//
TEST_CASE(simplify_add_debug_symbols)
{
    migraphx::module m1;
    {
        m1.set_use_debug_symbols();
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), x, one);
        sum1->add_debug_symbols({"onnx:add1"});
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), y, two);
        sum2->add_debug_symbols({"onnx:add2"});
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        sum3->add_debug_symbols({"onnx:add0"});
        m1.add_instruction(pass_op{}, sum3);
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        m2.set_use_debug_symbols();
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        sum1->add_debug_symbols({"onnx:add0", "onnx:add1", "onnx:add2"});
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), x, y);
        sum2->add_debug_symbols({"onnx:add0", "onnx:add1", "onnx:add2"});
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum2, sum1);
        sum3->add_debug_symbols({"onnx:add0", "onnx:add1", "onnx:add2"});
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(to_string(m1.sort()) == to_string(m2.sort()));
}

// Tests the replace_instruction(ins, rep) overload via find_unit_ops which
// simplifies add(relu(x), broadcast(0)) to relu(x).
//
//  Before:                       After:
//
//    x     0                       x
//    |     |                       |
//  relu   bcast                  relu  {add, relu}
//  {relu} (0.0)                    |
//     \   /                       pass
//     add  {add}
//      |
//     pass
//
TEST_CASE(replace_with_insref_debug_symbols)
{
    migraphx::module m1;
    {
        m1.set_use_debug_symbols();
        auto x    = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto zero = m1.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
        auto bcast =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), zero);
        auto relu_x = m1.add_instruction(migraphx::make_op("relu"), x);
        relu_x->add_debug_symbols({"onnx:relu"});
        auto add_r = m1.add_instruction(migraphx::make_op("add"), relu_x, bcast);
        add_r->add_debug_symbols({"onnx:add"});
        m1.add_instruction(pass_op{}, add_r);
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        m2.set_use_debug_symbols();
        auto x      = m2.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto relu_x = m2.add_instruction(migraphx::make_op("relu"), x);
        relu_x->add_debug_symbols({"onnx:add", "onnx:relu"});
        m2.add_instruction(pass_op{}, relu_x);
    }
    EXPECT(to_string(m1.sort()) == to_string(m2.sort()));
}

// Tests that debug_symbols propagate through the dead-code chain.
//
//  Before:                       After:
//
//    x                             x   y
//    |                              \ /
//   relu  {relu}                   mul  {add, relu}
//    |  y                            |
//    | /                            pass
//   add   {add}
//    |          (relu becomes dead code, removed by DCE)
//   pass
//
TEST_CASE(gather_replace_chain_debug_symbols)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        m1.set_use_debug_symbols();
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto relu_x = m1.add_instruction(migraphx::make_op("relu"), x);
        relu_x->add_debug_symbols({"onnx:relu"});
        auto add_r = m1.add_instruction(migraphx::make_op("add"), relu_x, y);
        add_r->add_debug_symbols({"onnx:add"});
        m1.add_instruction(pass_op{}, add_r);

        auto mul_r = m1.insert_instruction(add_r, migraphx::make_op("mul"), x, y);
        m1.replace_instruction(add_r, mul_r);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        m2.set_use_debug_symbols();
        auto x     = m2.add_parameter("x", s);
        auto y     = m2.add_parameter("y", s);
        auto mul_r = m2.add_instruction(migraphx::make_op("mul"), x, y);
        mul_r->add_debug_symbols({"onnx:add", "onnx:relu"});
        m2.add_instruction(pass_op{}, mul_r);
    }
    EXPECT(to_string(m1.sort()) == to_string(m2.sort()));
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
//    mul  {mul}                              |
//     |                                    pass
//    pass
//
TEST_CASE(simplify_mul_add_debug_symbols)
{
    migraphx::module m1;
    {
        m1.set_use_debug_symbols();
        auto x   = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one = m1.add_literal(3);
        auto two = m1.add_literal(2);
        auto sum = m1.add_instruction(migraphx::make_op("add"), one, x);
        sum->add_debug_symbols({"onnx:add"});
        auto mul = m1.add_instruction(migraphx::make_op("mul"), sum, two);
        mul->add_debug_symbols({"onnx:mul"});
        m1.add_instruction(pass_op{}, mul);
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        m2.set_use_debug_symbols();
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(3);
        auto two  = m2.add_literal(2);
        auto mul1 = m2.add_instruction(migraphx::make_op("mul"), two, x);
        mul1->add_debug_symbols({"onnx:add", "onnx:mul"});
        auto mul2 = m2.add_instruction(migraphx::make_op("mul"), two, one);
        mul2->add_debug_symbols({"onnx:add", "onnx:mul"});
        auto sum = m2.add_instruction(migraphx::make_op("add"), mul1, mul2);
        sum->add_debug_symbols({"onnx:add", "onnx:mul"});
        m2.add_instruction(pass_op{}, sum);
    }
    EXPECT(to_string(m1.sort()) == to_string(m2.sort()));
}

// Tests symbol propagation through find_div_const in simplify_algebra:
// div(x, c) -> mul(x, recip(c)).
//
//  Before:                       After:
//
//   x   c                         c
//    \ /                          |
//   div  {div}                  recip  {div}
//    |                           x  |
//   pass                          \ |
//                                 mul  {div}
//                                  |
//                                 pass
//
TEST_CASE(simplify_div_const_debug_symbols)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        m1.set_use_debug_symbols();
        auto x = m1.add_parameter("x", s);
        auto c =
            m1.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2, 3}},
                                             {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}});
        auto div_r = m1.add_instruction(migraphx::make_op("div"), x, c);
        div_r->add_debug_symbols({"onnx:div"});
        m1.add_instruction(pass_op{}, div_r);
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        m2.set_use_debug_symbols();
        auto x = m2.add_parameter("x", s);
        auto c =
            m2.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2, 3}},
                                             {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}});
        auto recip = m2.add_instruction(migraphx::make_op("recip"), c);
        recip->add_debug_symbols({"onnx:div"});
        auto mul_r = m2.add_instruction(migraphx::make_op("mul"), x, recip);
        mul_r->add_debug_symbols({"onnx:div"});
        m2.add_instruction(pass_op{}, mul_r);
    }
    EXPECT(to_string(m1.sort()) == to_string(m2.sort()));
}

// Unit test for the scoped_debug_symbols RAII guard's save/restore behavior.
// An outer guard sets "outer", a nested inner guard temporarily replaces it
// with "inner", and after the inner guard's destructor runs the outer symbols
// are restored. Instructions created in each scope carry only the symbols
// active at the time they were added.
//
//   scope     active symbols     instruction     gets symbol
//   -------   ----------------   -----------     -----------
//   outer     {"outer"}          add1(x, y)      {"outer"}
//     inner   {"inner"}          add2(add1, x)   {"inner"}
//   outer     {"outer"}          add3(add2, y)   {"outer"}
//
TEST_CASE(scoped_debug_symbols_nesting)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m("test", true);
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);

    migraphx::instruction_ref add1;
    migraphx::instruction_ref add2;
    migraphx::instruction_ref add3;
    {
        migraphx::scoped_debug_symbols outer(m, {"outer"});
        add1 = m.add_instruction(migraphx::make_op("add"), x, y);
        {
            migraphx::scoped_debug_symbols inner(m, {"inner"});
            add2 = m.add_instruction(migraphx::make_op("add"), add1, x);
        }
        add3 = m.add_instruction(migraphx::make_op("add"), add2, y);
    }

    EXPECT(add1->get_debug_symbols() == std::set<std::string>{"outer"});
    EXPECT(add2->get_debug_symbols() == std::set<std::string>{"inner"});
    EXPECT(add3->get_debug_symbols() == std::set<std::string>{"outer"});
}

// Three sequential adds fused into a single pointwise op via fuse_pointwise.
// All three ONNX node symbols should appear on the fused pointwise instruction.
// Extends pw_double_add to a longer chain.
//
//  Before:                         After:
//
//   x   y                          x  y  z  w
//    \ /                             \ | | /
//    add  {add1}                   pointwise  {add1, add2, add3}
//     |  z                              |
//     | /                            @return
//    add  {add2}
//     |  w
//     | /
//    add  {add3}
//     |
//   @return
//
TEST_CASE(pw_triple_add_fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        mm->set_use_debug_symbols();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto w    = mm->add_parameter("w", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        add1->add_debug_symbols({"onnx:add1"});
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        add2->add_debug_symbols({"onnx:add2"});
        auto add3 = mm->add_instruction(migraphx::make_op("add"), add2, w);
        add3->add_debug_symbols({"onnx:add3"});
        mm->add_return({add3});
    }
    migraphx::run_passes(p1, {migraphx::fuse_pointwise{}, migraphx::dead_code_elimination{}});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        mm->set_use_debug_symbols();
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", s);
        auto z = mm->add_parameter("z", s);
        auto w = mm->add_parameter("w", s);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x, y, z, w}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                auto add2 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
                return pm->add_instruction(migraphx::make_op("add"), add2, inputs[3]);
            });
        fadd->add_debug_symbols({"onnx:add1", "onnx:add2", "onnx:add3"});
        mm->add_return({fadd});
    }
    EXPECT(to_string(p1) == to_string(p2));
}

// Same add-reassociation pattern as simplify_add_debug_symbols but with
// set_use_debug_symbols() NOT called.
//
//  Before:                          After (flag OFF):
//
//   x   1    y   2                    1   2       x   y
//    \ /      \ /                      \ /         \ /
//   add1{a1} add2{a2}                 add{}     add{}
//      \     /                           \     /
//      add0{a0}                           add{a0}
//        |                                  |
//       pass                              pass
//
//  (compare with simplify_add_debug_symbols where flag ON
//   gives {a0, a1, a2} on every instruction)
//
TEST_CASE(no_propagation_without_flag)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m1.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m1.add_literal(1);
        auto two  = m1.add_literal(2);
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), x, one);
        sum1->add_debug_symbols({"onnx:add1"});
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), y, two);
        sum2->add_debug_symbols({"onnx:add2"});
        auto sum3 = m1.add_instruction(migraphx::make_op("add"), sum1, sum2);
        sum3->add_debug_symbols({"onnx:add0"});
        m1.add_instruction(pass_op{}, sum3);
    }
    migraphx::run_passes(m1, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::int32_type, {1}});
        auto y    = m2.add_parameter("y", {migraphx::shape::int32_type, {1}});
        auto one  = m2.add_literal(1);
        auto two  = m2.add_literal(2);
        auto sum1 = m2.add_instruction(migraphx::make_op("add"), one, two);
        auto sum2 = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto sum3 = m2.add_instruction(migraphx::make_op("add"), sum2, sum1);
        sum3->add_debug_symbols({"onnx:add0"});
        m2.add_instruction(pass_op{}, sum3);
    }
    EXPECT(to_string(m1.sort()) == to_string(m2.sort()));
}

// Verifies that debug symbols appear in the module's printed/serialized
// output using the expected comment format produced by instruction::print.
//
//  Printed output includes:
//    @2 = add(@0,@1) -> float_type, {2, 3} /* sym_a, sym_b */
//
TEST_CASE(debug_symbols_in_print)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m;
    m.set_use_debug_symbols();
    auto x   = m.add_parameter("x", s);
    auto y   = m.add_parameter("y", s);
    auto add = m.add_instruction(migraphx::make_op("add"), x, y);
    add->add_debug_symbols({"sym_a", "sym_b"});
    m.add_instruction(pass_op{}, add);

    auto str = migraphx::to_string(m);
    EXPECT(str.find("/* sym_a, sym_b */") != std::string::npos);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
