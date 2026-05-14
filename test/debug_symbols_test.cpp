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
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/stringutils.hpp>

#include <test.hpp>

// Two adds replaced by a single pass_op via replace_instruction,
// emulating what fuse_pointwise does without running the pass.
// add1 feeds only into add2 (splice chain), so both symbols merge.
//
//  Before:                          After:
//
//   x   y                           x  y  z
//    \ /                              \ | /
//    add  {add1}                    pass  {add1, add2}
//     |  z                             |
//     | /                           @return
//    add  {add2}
//     |
//   @return
//
TEST_CASE(pw_double_add)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto y    = m1.add_parameter("y", s);
        auto z    = m1.add_parameter("z", s);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_debug_symbols(add1, {"add1"});
        auto add2 = m1.add_instruction(migraphx::make_op("add"), add1, z);
        m1.add_debug_symbols(add2, {"add2"});
        m1.add_return({add2});

        m1.replace_instruction(add2, pass_op{}, x, y, z);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto y    = m2.add_parameter("y", s);
        auto z    = m2.add_parameter("z", s);
        auto fadd = m2.add_instruction(pass_op{}, x, y, z);
        m2.add_debug_symbols(fadd, {"add1", "add2"});
        m2.add_return({fadd});
    }
    EXPECT(m1 == m2);
}

// Diamond of four adds replaced by a single pass_op via replace_instruction,
// emulating what fuse_pointwise does without running the pass.
// add2 and add3 each feed only into add4 (splice chain), and add1
// feeds only into add2 and add3 (both in the chain), so all four
// symbols merge onto the replacement.
//
//  Before:                       After:
//
//    x   y                        x   y
//     \ /                          \ /
//     add1 {add1}               pass  {add1, add2, add3, add4}
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
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto y    = m1.add_parameter("y", s);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_debug_symbols(add1, {"onnx:add1"});
        auto add2 = m1.add_instruction(migraphx::make_op("add"), add1, x);
        m1.add_debug_symbols(add2, {"onnx:add2"});
        auto add3 = m1.add_instruction(migraphx::make_op("add"), add1, y);
        m1.add_debug_symbols(add3, {"onnx:add3"});
        auto add4 = m1.add_instruction(migraphx::make_op("add"), add2, add3);
        m1.add_debug_symbols(add4, {"onnx:add4"});
        m1.add_return({add4});

        m1.replace_instruction(add4, pass_op{}, x, y);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto y    = m2.add_parameter("y", s);
        auto fadd = m2.add_instruction(pass_op{}, x, y);
        m2.add_debug_symbols(fadd, {"onnx:add1", "onnx:add2", "onnx:add3", "onnx:add4"});
        m2.add_return({fadd});
    }
    EXPECT(m1 == m2);
}

// Debug symbols should not propagate above the fusion boundary.
// The dot is a common ancestor in both the old and new splice, so
// its symbol stays on the dot instruction. Only add1 and add2 (the
// splice chain between dot and the replacement) merge their symbols.
//
//  Before:                          After:
//
//   x   a                            x    a
//    \ /                              \ /
//    dot  {gemm1}                    dot  {gemm1}
//     |  y                          /
//     | /                          /  y    z
//    add  {add1}                  \  |  /
//     |  z                       pass  {add1, add2}
//     | /                            |
//    add  {add2}                  @return
//     |
//   @return
//
TEST_CASE(gemm_add_add)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto y    = m1.add_parameter("y", s1);
        auto z    = m1.add_parameter("z", s1);
        auto a    = m1.add_literal(migraphx::generate_literal(s2, 0));
        auto gemm = m1.add_instruction(migraphx::make_op("dot"), x, a);
        m1.add_debug_symbols(gemm, {"gemm1"});
        auto add1 = m1.add_instruction(migraphx::make_op("add"), gemm, y);
        m1.add_debug_symbols(add1, {"add1"});
        auto add2 = m1.add_instruction(migraphx::make_op("add"), add1, z);
        m1.add_debug_symbols(add2, {"add2"});
        m1.add_return({add2});

        m1.replace_instruction(add2, pass_op{}, gemm, y, z);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s1);
        auto y    = m2.add_parameter("y", s1);
        auto z    = m2.add_parameter("z", s1);
        auto a    = m2.add_literal(migraphx::generate_literal(s2, 0));
        auto gemm = m2.add_instruction(migraphx::make_op("dot"), x, a);
        m2.add_debug_symbols(gemm, {"gemm1"});
        auto fadd = m2.add_instruction(pass_op{}, gemm, y, z);
        m2.add_debug_symbols(fadd, {"add1", "add2"});
        m2.add_return({fadd});
    }
    EXPECT(m1 == m2);
}

// Horizontal fusion of two dot ops sharing the same input, emulating
// what simplify_algebra does via insert_instruction + batch_replace_instruction.
// The first replacement sees fused_dot with a single output (the second
// dot hasn't been replaced yet), so the new splice chain traverses
// through fused_dot and concat. All four new instructions receive the
// merged {gemm1, gemm2} symbols.
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
//                         slice {g1, g2}  slice {g1, g2}
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

        auto concat    = m1.insert_instruction(x, migraphx::make_op("concat", {{"axis", 2}}), a, b);
        auto fused_dot = m1.insert_instruction(x, migraphx::make_op("dot"), input, concat);
        m1.batch_replace_instruction({
            {x,
             migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}),
             {fused_dot},
             {}},
            {y,
             migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {4}}}),
             {fused_dot},
             {}},
        });
    }

    migraphx::module m2;
    {
        auto input  = m2.add_parameter("input", s);
        auto a      = m2.add_literal(migraphx::generate_literal(s, 0));
        auto b      = m2.add_literal(migraphx::generate_literal(s, 1));
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), a, b);
        m2.add_debug_symbols(concat, {"gemm1", "gemm2"});
        auto fused_dot = m2.add_instruction(migraphx::make_op("dot"), input, concat);
        m2.add_debug_symbols(fused_dot, {"gemm1", "gemm2"});
        auto sx = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), fused_dot);
        m2.add_debug_symbols(sx, {"gemm1", "gemm2"});
        auto sy = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {4}}}), fused_dot);
        m2.add_debug_symbols(sy, {"gemm1", "gemm2"});
        auto sum = m2.add_instruction(migraphx::make_op("add"), sx, sy);
        m2.add_debug_symbols(sum, {"sum"});
        m2.add_return({sum});
    }
    EXPECT(m1 == m2);
}

// Emulates the find_pointwise_reduce fusion using pass_op + replace_instruction.
// A pass_op standing in for the pointwise feeds only into a pass_op standing
// in for the reduce (splice chain), so replace_instruction merges both symbols.
//
//  Before:                                        After:
//
//   x   y                                          x   y
//    \ /                                            \ /
//    add  {add0}                                   add  {add0}
//     |                                             |
//    relu  {relu0}                                 relu  {relu0}
//     |  z                                          |  z
//     | /                                           | /
//    pass  {pointwise}                             pass  {pointwise, reduce_sum}
//     |                                             |
//    pass  {reduce_sum}                            relu  {relu1}
//     |                                             |
//    relu  {relu1}                               @return
//     |
//   @return
//
TEST_CASE(pointwise_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto y    = m1.add_parameter("y", s);
        auto z    = m1.add_parameter("z", s);
        auto curr = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_debug_symbols(curr, {"add0"});
        curr = m1.add_instruction(migraphx::make_op("relu"), curr);
        m1.add_debug_symbols(curr, {"relu0"});
        auto pw = m1.add_instruction(pass_op{}, curr, z);
        m1.add_debug_symbols(pw, {"pointwise"});
        auto rs = m1.add_instruction(pass_op{}, pw);
        m1.add_debug_symbols(rs, {"reduce_sum"});
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), rs);
        m1.add_debug_symbols(relu1, {"relu1"});
        m1.add_return({relu1});

        m1.replace_instruction(rs, pass_op{}, curr, z);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto y    = m2.add_parameter("y", s);
        auto z    = m2.add_parameter("z", s);
        auto curr = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_debug_symbols(curr, {"add0"});
        curr = m2.add_instruction(migraphx::make_op("relu"), curr);
        m2.add_debug_symbols(curr, {"relu0"});
        auto fused = m2.add_instruction(pass_op{}, curr, z);
        m2.add_debug_symbols(fused, {"pointwise", "reduce_sum"});
        auto relu1 = m2.add_instruction(migraphx::make_op("relu"), fused);
        m2.add_debug_symbols(relu1, {"relu1"});
        m2.add_return({relu1});
    }
    EXPECT(m1 == m2);
}

// Tests symbol propagation through add reassociation, emulating
// find_double_add_lit_broadcast via insert_instruction + replace_instruction.
// add(add(x,1), add(y,2)) -> add(add(x,y), add(1,2)).
// sum1 and sum2 each feed only into sum3 (splice chain), so all three
// symbols merge onto every instruction in the new splice.
//
//  Before:                          After:
//
//   x   1    y   2                    1   2       x   y
//    \ /      \ /                      \ /         \ /
//   add1{a1} add2{a2}                 add{a0,a1,a2} add{a0,a1,a2}
//      \     /                              \       /
//      add0{a0}                             add{a0,a1,a2}
//
TEST_CASE(simplify_add)
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

        auto sumab = m1.insert_instruction(sum3, migraphx::make_op("add"), one, two);
        auto sumxy = m1.insert_instruction(sum3, migraphx::make_op("add"), x, y);
        m1.replace_instruction(sum3, migraphx::make_op("add"), sumxy, sumab);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

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

// Tests the replace_instruction(ins, rep) overload directly, emulating
// what find_unit_ops does: add(relu(x), broadcast(0)) -> relu(x).
// replace_instruction(add_r, relu_x) redirects add's outputs to relu,
// and relu inherits add's {onnx:add} symbol.
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
TEST_CASE(replace_with_insref)
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

        m1.replace_instruction(add_r, relu_x);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

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
TEST_CASE(gather_replace_chain)
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

// Tests the distributive law transform via insert_instruction + replace_instruction,
// emulating find_mul_add: mul(add(3, x), 2) -> add(mul(2, x), mul(2, 3)).
// add feeds only into mul (splice chain), so both symbols merge onto
// every instruction in the new splice.
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
TEST_CASE(simplify_mul_add)
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

        auto ax = m1.insert_instruction(mul, migraphx::make_op("mul"), two, x);
        auto ab = m1.insert_instruction(mul, migraphx::make_op("mul"), two, one);
        m1.replace_instruction(mul, migraphx::make_op("add"), ax, ab);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

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

// Tests symbol propagation through insert_instruction + replace_instruction,
// emulating find_div_const: div(x, c) -> mul(x, recip(c)).
// div is the only instruction in the splice chain, so its {onnx:div} symbol
// propagates to both recip and mul.
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
TEST_CASE(simplify_div_const)
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

        auto recip = m1.insert_instruction(div_r, migraphx::make_op("recip"), c);
        m1.replace_instruction(div_r, migraphx::make_op("mul"), x, recip);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

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
//    @2 = add(@0,@1) -> float_type, {2, 3} # sym_a, sym_b
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
    EXPECT(str.find("# sym_a, sym_b") != std::string::npos);
}

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

// Emulates rewrite_nearest_resize: resize(x) -> gather(reshape(x), indices).
// resize is the only symbolized instruction in the splice chain, so
// {onnx:resize} propagates to both reshape and gather. The literal
// (gather indices) does not receive symbols.
//
//  Before:                          After:
//
//    x                                x
//    |                                |
//  resize {resize}               reshape {resize}   indices
//                                     \             /
//                                     gather {resize}
//
TEST_CASE(rewrite_resize_debug_symbols)
{
    migraphx::shape in_s{migraphx::shape::float_type, {1, 1, 2, 2}};
    // clang-format off
    std::vector<int> indices = {0, 0, 1, 1,
                                0, 0, 1, 1,
                                2, 2, 3, 3,
                                2, 2, 3, 3};
    // clang-format on

    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", in_s);
        auto resize = m1.add_instruction(
            migraphx::make_op("resize",
                              {{"scales", {1.0f, 1.0f, 2.0f, 2.0f}},
                               {"nearest_mode", "floor"},
                               {"coordinate_transformation_mode", "asymmetric"}}),
            x);
        m1.add_debug_symbols(resize, {"onnx:resize"});
        m1.add_return({resize});

        auto rsp = m1.insert_instruction(resize, migraphx::make_op("reshape", {{"dims", {4}}}), x);
        auto ins_ind = m1.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1, 1, 4, 4}}, indices});
        m1.replace_instruction(resize, migraphx::make_op("gather", {{"axis", 0}}), rsp, ins_ind);
    }
    migraphx::run_passes(m1, {migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", in_s);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), x);
        m2.add_debug_symbols(rsp, {"onnx:resize"});
        auto ins_ind = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1, 1, 4, 4}}, indices});
        auto gather = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp, ins_ind);
        m2.add_debug_symbols(gather, {"onnx:resize"});
        m2.add_return({gather});
    }
    EXPECT(m1.sort() == m2.sort());
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
