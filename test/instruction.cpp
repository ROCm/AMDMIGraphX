//
// The MIT License (MIT)
//
// Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include "test.hpp"

TEST_CASE(check_undefined)
{
    migraphx::module m;
    auto und = m.add_instruction(migraphx::make_op("undefined"));
    auto cov = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), und);
    auto abs = m.add_instruction(migraphx::make_op("abs"), cov);

    migraphx::shape xs{migraphx::shape::float_type, {2, 3}};
    std::vector<float> datax = {1, 2, 3, 4, 5, 6};

    auto lit = m.add_literal(migraphx::literal(xs, datax));
    auto mul = m.add_instruction(migraphx::make_op("mul"), lit, lit);

    EXPECT(und->is_undefined());
    EXPECT(cov->is_undefined());
    EXPECT(abs->is_undefined());
    EXPECT(not lit->is_undefined());
    EXPECT(not mul->is_undefined());
}

TEST_CASE(check_replace_shape)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 2}};
    auto input  = m.add_parameter("x", s);
    auto reduce = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), input);
    auto abs    = m.add_instruction(migraphx::make_op("abs"), reduce);
    auto sin    = m.add_instruction(migraphx::make_op("sin"), reduce);
    auto add    = m.add_instruction(migraphx::make_op("add"), abs, sin);

    reduce->replace(migraphx::make_op("reduce_sum", {{"axes", {1}}}));

    migraphx::shape r{migraphx::shape::float_type, {3, 1}};
    EXPECT(reduce->get_shape() == r);
    EXPECT(abs->get_shape() == r);
    EXPECT(sin->get_shape() == r);
    EXPECT(add->get_shape() == r);
}

TEST_CASE(check_replace_dag)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 2}};
    auto input  = m.add_parameter("x", s);
    auto reduce = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), input);
    auto abs    = m.add_instruction(migraphx::make_op("abs"), reduce);
    auto sin    = m.add_instruction(migraphx::make_op("sin"), reduce);
    auto add    = m.add_instruction(migraphx::make_op("add"), abs, sin);
    auto add2   = m.add_instruction(migraphx::make_op("add"), add, reduce);

    reduce->replace(migraphx::make_op("reduce_sum", {{"axes", {1}}}));

    migraphx::shape r{migraphx::shape::float_type, {3, 1}};
    EXPECT(reduce->get_shape() == r);
    EXPECT(abs->get_shape() == r);
    EXPECT(sin->get_shape() == r);
    EXPECT(add->get_shape() == r);
    EXPECT(add2->get_shape() == r);
}

// Tests for the reaches function
//
// Linear graph:
//
// x --> relu --> tanh --> abs
//
TEST_CASE(reaches_direct_connection)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x    = m.add_parameter("x", s);
    auto relu = m.add_instruction(migraphx::make_op("relu"), x);
    auto tanh = m.add_instruction(migraphx::make_op("tanh"), relu);
    auto abs  = m.add_instruction(migraphx::make_op("abs"), tanh);

    // Direct connections
    EXPECT(migraphx::reaches(x, relu, &m));
    EXPECT(migraphx::reaches(relu, tanh, &m));
    EXPECT(migraphx::reaches(tanh, abs, &m));

    // Transitive connections
    EXPECT(migraphx::reaches(x, tanh, &m));
    EXPECT(migraphx::reaches(x, abs, &m));
    EXPECT(migraphx::reaches(relu, abs, &m));

    // Same instruction
    EXPECT(migraphx::reaches(x, x, &m));
    EXPECT(migraphx::reaches(abs, abs, &m));
}

//
// Branched graph:
//
//     x       y
//      \     /
//       \   /
//        v v
//        add
//       /   \
//      v     v
//   relu1   relu2
//      \     /
//       \   /
//        v v
//      concat
//
TEST_CASE(reaches_branched_connections)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x      = m.add_parameter("x", s);
    auto y      = m.add_parameter("y", s);
    auto add    = m.add_instruction(migraphx::make_op("add"), x, y);
    auto relu1  = m.add_instruction(migraphx::make_op("relu"), add);
    auto relu2  = m.add_instruction(migraphx::make_op("relu"), add);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), relu1, relu2);

    // Branch connections
    EXPECT(migraphx::reaches(x, add, &m));
    EXPECT(migraphx::reaches(y, add, &m));
    EXPECT(migraphx::reaches(add, relu1, &m));
    EXPECT(migraphx::reaches(add, relu2, &m));
    EXPECT(migraphx::reaches(relu1, concat, &m));
    EXPECT(migraphx::reaches(relu2, concat, &m));

    // Transitive connections
    EXPECT(migraphx::reaches(x, relu1, &m));
    EXPECT(migraphx::reaches(x, relu2, &m));
    EXPECT(migraphx::reaches(y, relu1, &m));
    EXPECT(migraphx::reaches(y, relu2, &m));
    EXPECT(migraphx::reaches(x, concat, &m));
    EXPECT(migraphx::reaches(y, concat, &m));

    // No connections
    EXPECT(not migraphx::reaches(relu1, relu2, &m));
}

//
// Complex diamond graph:
//
//     x       y
//      \     /
//       \   /
//        v v
//        add
//       /   \
//      v     v
//   relu    tanh
//     |      |
//     v      v
//    abs     sin
//      \    /
//       \  /
//        vv
//      concat
//
TEST_CASE(reaches_complex_graph1)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x      = m.add_parameter("x", s);
    auto y      = m.add_parameter("y", s);
    auto add    = m.add_instruction(migraphx::make_op("add"), x, y);
    auto relu   = m.add_instruction(migraphx::make_op("relu"), add);
    auto tanh   = m.add_instruction(migraphx::make_op("tanh"), add);
    auto abs    = m.add_instruction(migraphx::make_op("abs"), relu);
    auto sin    = m.add_instruction(migraphx::make_op("sin"), tanh);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), abs, sin);

    // Test complex paths
    EXPECT(migraphx::reaches(x, concat, &m));
    EXPECT(migraphx::reaches(y, concat, &m));
    EXPECT(migraphx::reaches(add, concat, &m));
    EXPECT(migraphx::reaches(relu, concat, &m));
    EXPECT(migraphx::reaches(tanh, concat, &m));
    EXPECT(migraphx::reaches(abs, concat, &m));
    EXPECT(migraphx::reaches(sin, concat, &m));

    // Test paths through different branches
    EXPECT(migraphx::reaches(x, abs, &m));
    EXPECT(migraphx::reaches(y, sin, &m));
    EXPECT(migraphx::reaches(add, abs, &m));
    EXPECT(migraphx::reaches(add, sin, &m));

    // Test non-existing paths
    EXPECT(not migraphx::reaches(relu, sin, &m));
    EXPECT(not migraphx::reaches(relu, tanh, &m));
    EXPECT(not migraphx::reaches(tanh, abs, &m));
    EXPECT(not migraphx::reaches(abs, sin, &m));
}

TEST_CASE(reaches_complex_graph2)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x      = m.add_parameter("x", s);
    auto y      = m.add_parameter("y", s);
    auto add    = m.add_instruction(migraphx::make_op("add"), x, y);
    auto relu   = m.add_instruction(migraphx::make_op("relu"), add);
    auto tanh   = m.add_instruction(migraphx::make_op("tanh"), add);
    auto abs    = m.add_instruction(migraphx::make_op("abs"), relu);
    auto sin    = m.add_instruction(migraphx::make_op("sin"), tanh);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), abs, sin);

    // Test complex paths
    EXPECT(migraphx::reaches(x, concat));
    EXPECT(migraphx::reaches(y, concat));
    EXPECT(migraphx::reaches(add, concat));
    EXPECT(migraphx::reaches(relu, concat));
    EXPECT(migraphx::reaches(tanh, concat));
    EXPECT(migraphx::reaches(abs, concat));
    EXPECT(migraphx::reaches(sin, concat));

    // Test paths through different branches
    EXPECT(migraphx::reaches(x, abs));
    EXPECT(migraphx::reaches(y, sin));
    EXPECT(migraphx::reaches(add, abs));
    EXPECT(migraphx::reaches(add, sin));

    // Test non-existing paths
    EXPECT(not migraphx::reaches(relu, sin));
    EXPECT(not migraphx::reaches(relu, tanh));
    EXPECT(not migraphx::reaches(tanh, abs));
    EXPECT(not migraphx::reaches(abs, sin));
}

// Tests for the is_interdependent function
//
// Linear chain:
//
// x --> relu --> tanh --> abs
//
TEST_CASE(is_interdependent_simple)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x    = m.add_parameter("x", s);
    auto relu = m.add_instruction(migraphx::make_op("relu"), x);
    auto tanh = m.add_instruction(migraphx::make_op("tanh"), relu);
    auto abs  = m.add_instruction(migraphx::make_op("abs"), tanh);

    // Sequential chain - these should be interdependent
    std::vector<migraphx::instruction_ref> seq_chain = {x, relu, tanh, abs};
    EXPECT(migraphx::is_interdependent(seq_chain, &m, m.begin()));

    // Subset of chain - also interdependent
    std::vector<migraphx::instruction_ref> sub_chain = {x, relu, abs};
    EXPECT(migraphx::is_interdependent(sub_chain, &m, m.begin()));

    // Single instruction is always interdependent
    std::vector<migraphx::instruction_ref> single = {x};
    EXPECT(migraphx::is_interdependent(single, &m, m.begin()));

    // Empty vector is also interdependent (vacuously true)
    std::vector<migraphx::instruction_ref> empty = {};
    EXPECT(migraphx::is_interdependent(empty, &m, m.begin()));
}

//
// Branched Y graph:
//
//     x       y
//      \     /
//       \   /
//        v v
//        add
//       /   \
//      v     v
//   relu    tanh
//
TEST_CASE(is_interdependent_branched)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x    = m.add_parameter("x", s);
    auto y    = m.add_parameter("y", s);
    auto add  = m.add_instruction(migraphx::make_op("add"), x, y);
    auto relu = m.add_instruction(migraphx::make_op("relu"), add);
    auto tanh = m.add_instruction(migraphx::make_op("tanh"), add);

    // Interdependent branches
    std::vector<migraphx::instruction_ref> interdep = {add, relu, tanh};
    EXPECT(migraphx::is_interdependent(interdep, &m, m.begin()));

    // Independent parameters
    std::vector<migraphx::instruction_ref> indep = {x, y};
    EXPECT(not migraphx::is_interdependent(indep, &m, m.begin()));

    // Mixed dependent and independent
    std::vector<migraphx::instruction_ref> mixed = {x, relu, tanh};
    EXPECT(migraphx::is_interdependent(mixed, &m, m.begin()));
}

//
// Complex graph with multiple paths:
//
//     x       y       z
//      \     / \     /
//       \   /   \   /
//        v v     v v
//       add1    add2
//        |       |
//        v       v
//      relu1   relu2
//        \     /
//         \   /
//          v v
//        concat
//
TEST_CASE(is_interdependent_complex)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x      = m.add_parameter("x", s);
    auto y      = m.add_parameter("y", s);
    auto z      = m.add_parameter("z", s);
    auto add1   = m.add_instruction(migraphx::make_op("add"), x, y);
    auto add2   = m.add_instruction(migraphx::make_op("add"), y, z);
    auto relu1  = m.add_instruction(migraphx::make_op("relu"), add1);
    auto relu2  = m.add_instruction(migraphx::make_op("relu"), add2);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), relu1, relu2);

    // Complex interdependent set
    std::vector<migraphx::instruction_ref> complex_dep = {y, add1, add2, relu1, relu2, concat};
    EXPECT(migraphx::is_interdependent(complex_dep, &m, m.begin()));

    // Independent branches
    std::vector<migraphx::instruction_ref> indep_branches = {add1, add2};
    EXPECT(not migraphx::is_interdependent(indep_branches, &m, m.begin()));

    // Independent outputs
    std::vector<migraphx::instruction_ref> indep_outputs = {relu1, relu2};
    EXPECT(not migraphx::is_interdependent(indep_outputs, &m, m.begin()));
}

//
// Long chain:
//
//   x     y
//    \   /
//     \ /
//      v
//     add --> relu1 --> relu2 --> ... --> relu9
//
TEST_CASE(is_interdependent_large_set)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);

    // Create a chain of 10 instructions
    auto curr = m.add_instruction(migraphx::make_op("add"), x, y);
    std::vector<migraphx::instruction_ref> chain = {curr};

    for(int i = 0; i < 9; i++)
    {
        curr = m.add_instruction(migraphx::make_op("relu"), curr);
        chain.push_back(curr);
    }

    // Test with large chain
    EXPECT(migraphx::is_interdependent(chain, &m, m.begin()));

    // Take a subset that skips some elements
    std::vector<migraphx::instruction_ref> subset = {
        chain[0], chain[2], chain[4], chain[6], chain[8]};
    EXPECT(migraphx::is_interdependent(subset, &m, m.begin()));
}

// Long chain (32 instructions between start and end):
//   x --> relu0 --> relu1 --> ... --> relu30 --> abs
TEST_CASE(reaches_large_linear)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x = m.add_parameter("x", s);

    const int chain_len = 31;
    std::vector<migraphx::instruction_ref> chain;
    chain.push_back(x);
    for(int i = 0; i < chain_len; i++)
        chain.push_back(m.add_instruction(migraphx::make_op("relu"), chain.back()));
    auto last = m.add_instruction(migraphx::make_op("abs"), chain.back());

    // Start to end (distance = 32)
    EXPECT(migraphx::reaches(x, last, &m));

    // Mid-chain to end
    EXPECT(migraphx::reaches(chain[15], last, &m));

    // Start to mid-chain
    EXPECT(migraphx::reaches(x, chain[20], &m));

    // Same instruction
    EXPECT(migraphx::reaches(chain[16], chain[16], &m));
}

//
// Two interleaved independent chains (no connection between them):
//
//   x --> relu0 --> relu1 --> ... --> relu19
//   y --> tanh0 --> tanh1 --> ... --> tanh19
//
TEST_CASE(reaches_large_independent_chains)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);

    const int chain_len = 20;
    std::vector<migraphx::instruction_ref> chain_a;
    chain_a.push_back(x);
    for(int i = 0; i < chain_len; i++)
        chain_a.push_back(m.add_instruction(migraphx::make_op("relu"), chain_a.back()));

    std::vector<migraphx::instruction_ref> chain_b;
    chain_b.push_back(y);
    for(int i = 0; i < chain_len; i++)
        chain_b.push_back(m.add_instruction(migraphx::make_op("tanh"), chain_b.back()));

    // Within each chain: reachable
    EXPECT(migraphx::reaches(x, chain_a.back(), &m));
    EXPECT(migraphx::reaches(y, chain_b.back(), &m));

    // Across chains: not reachable
    EXPECT(not migraphx::reaches(x, chain_b.back(), &m));
    EXPECT(not migraphx::reaches(y, chain_a.back(), &m));
}

// Tests for the find_instructions_between function
//
// Linear chain:
//
// x --> relu --> tanh --> abs
//
TEST_CASE(find_instructions_between_simple)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x    = m.add_parameter("x", s);
    auto relu = m.add_instruction(migraphx::make_op("relu"), x);
    auto tanh = m.add_instruction(migraphx::make_op("tanh"), relu);
    auto abs  = m.add_instruction(migraphx::make_op("abs"), tanh);

    // Find instructions between x and abs
    auto result = migraphx::find_instructions_between(x, abs, &m);

    // Should include x, relu, tanh, abs
    EXPECT(result.count(x) == 1);
    EXPECT(result.count(relu) == 1);
    EXPECT(result.count(tanh) == 1);
    EXPECT(result.count(abs) == 1);
    EXPECT(result.size() == 4);

    // Find instructions between relu and abs
    auto result2 = migraphx::find_instructions_between(relu, abs, &m);

    // Should include relu, tanh, abs
    EXPECT(result2.count(x) == 0);
    EXPECT(result2.count(relu) == 1);
    EXPECT(result2.count(tanh) == 1);
    EXPECT(result2.count(abs) == 1);
    EXPECT(result2.size() == 3);
}

//
// Branched Y graph:
//
//     x       y
//      \     /
//       \   /
//        v v
//        add
//       /   \
//      v     v
//   relu    tanh
//      \     /
//       \   /
//        v v
//      concat
//
TEST_CASE(find_instructions_between_branched)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto x      = m.add_parameter("x", s);
    auto y      = m.add_parameter("y", s);
    auto add    = m.add_instruction(migraphx::make_op("add"), x, y);
    auto relu   = m.add_instruction(migraphx::make_op("relu"), add);
    auto tanh   = m.add_instruction(migraphx::make_op("tanh"), add);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), relu, tanh);

    // Find instructions between add and concat
    auto result = migraphx::find_instructions_between(add, concat, &m);

    // Should include add, relu, tanh, concat
    EXPECT(result.count(add) == 1);
    EXPECT(result.count(relu) == 1);
    EXPECT(result.count(tanh) == 1);
    EXPECT(result.count(concat) == 1);
    EXPECT(result.size() == 4);

    // Find instructions between x and concat
    auto result2 = migraphx::find_instructions_between(x, concat, &m);

    // Should include x, add, relu, tanh, concat but not y
    EXPECT(result2.count(x) == 1);
    EXPECT(result2.count(y) == 0);
    EXPECT(result2.count(add) == 1);
    EXPECT(result2.count(relu) == 1);
    EXPECT(result2.count(tanh) == 1);
    EXPECT(result2.count(concat) == 1);
    EXPECT(result2.size() == 5);
}

//
// Complex diamond graph with multiple inputs:
//
//    w     x     y     z
//     \   /      \   /
//      \ /        \ /
//       v          v
//     add1       add2
//       \         /
//        \       /
//         \     /
//          \   /
//           v v
//           mul
//          /   \
//         v     v
//       relu   tanh
//         \     /
//          \   /
//           v v
//         concat
//
TEST_CASE(find_instructions_between_complex)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto w = m.add_parameter("w", s);
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto z = m.add_parameter("z", s);

    auto add1   = m.add_instruction(migraphx::make_op("add"), w, x);
    auto add2   = m.add_instruction(migraphx::make_op("add"), y, z);
    auto mul    = m.add_instruction(migraphx::make_op("mul"), add1, add2);
    auto relu   = m.add_instruction(migraphx::make_op("relu"), mul);
    auto tanh   = m.add_instruction(migraphx::make_op("tanh"), mul);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), relu, tanh);

    // Find instructions between w and concat
    auto result = migraphx::find_instructions_between(w, concat, &m);

    // Should include w, add1, mul, relu, tanh, concat but not x, y, z, add2
    EXPECT(result.count(w) == 1);
    EXPECT(result.count(add1) == 1);
    EXPECT(result.count(mul) == 1);
    EXPECT(result.count(relu) == 1);
    EXPECT(result.count(tanh) == 1);
    EXPECT(result.count(concat) == 1);
    EXPECT(result.size() == 6);

    EXPECT(result.count(x) == 0);
    EXPECT(result.count(y) == 0);
    EXPECT(result.count(z) == 0);
    EXPECT(result.count(add2) == 0);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
