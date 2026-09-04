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

#include <migraphx/gpu/mlir.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <test.hpp>

struct dot_graph
{
    migraphx::instruction_ref dot;
    migraphx::instruction_ref bias;
};

static dot_graph make_dot_graph(migraphx::module& m)
{
    auto a    = m.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {1, 5, 4}});
    auto b    = m.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {1, 4, 3}});
    auto bias = m.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {1, 5, 3}});
    return {m.add_instruction(migraphx::make_op("dot"), a, b), bias};
}

TEST_CASE(find_final_split_before_pointwise)
{
    migraphx::module m;
    auto graph = make_dot_graph(m);
    auto add   = m.add_instruction(migraphx::make_op("add"), graph.dot, graph.bias);
    auto tanh  = m.add_instruction(migraphx::make_op("tanh"), add);
    m.add_return({tanh});

    EXPECT(migraphx::gpu::find_final_split(graph.dot) == add);
}

TEST_CASE(find_final_split_with_multiple_outputs)
{
    migraphx::module m;
    auto graph = make_dot_graph(m);
    auto add   = m.add_instruction(migraphx::make_op("add"), graph.dot, graph.bias);
    auto mul   = m.add_instruction(migraphx::make_op("mul"), graph.dot, graph.bias);
    m.add_return({add, mul});

    EXPECT(migraphx::gpu::find_final_split(graph.dot) == graph.dot);
}

TEST_CASE(find_final_split_without_boundary)
{
    migraphx::module m;
    auto graph   = make_dot_graph(m);
    auto reshape = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 15}}}), graph.dot);

    EXPECT(migraphx::gpu::find_final_split(graph.dot) == reshape);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
