/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/propagate_reshape_layout.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>
#include "make_precompile_op.hpp"

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m, {migraphx::gpu::propagate_reshape_layout{}, migraphx::dead_code_elimination{}});
}

// A reshape_lazy that cannot alias its non-standard input keeps a standardizing
// gpu::contiguous after eliminate_contiguous. The pass should replace it with a layout
// that repacks the input into the order reshape_lazy needs to propagate the permutation.
TEST_CASE(propagate_permutation)
{
    migraphx::shape in{migraphx::shape::float_type, {1, 1, 1024, 1024}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", in);
        auto r =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 256, 4, 256, 4}}}), x);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 4, 1, 3}}}), r);
        // post-eliminate_contiguous state: a standardizing gpu::contiguous feeds reshape_lazy
        auto alloc = m1.add_instruction(
            migraphx::make_op("allocate",
                              {{"shape",
                                migraphx::to_value(migraphx::shape{migraphx::shape::float_type,
                                                                   t->get_shape().lens()})}}));
        auto c = m1.add_instruction(migraphx::make_op("gpu::contiguous"), t, alloc);
        auto rl =
            m1.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 16, 256, 256}}}), c);
        m1.add_return({rl});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", in);
        auto r =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 256, 4, 256, 4}}}), x);
        auto t = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 4, 1, 3}}}), r);
        // layout repacks the transpose into the packed memory order reshape_lazy can alias
        auto l_shape = migraphx::shape::from_permutation(
            migraphx::shape::float_type, {1, 4, 4, 256, 256}, {0, 3, 4, 1, 2});
        auto alloc = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(l_shape)}}));
        auto layout = m2.add_instruction(
            make_precompile_op(migraphx::make_op("layout", {{"permutation", {0, 3, 4, 1, 2}}})),
            t,
            alloc);
        auto rl = m2.add_instruction(
            migraphx::make_op("reshape_lazy", {{"dims", {1, 16, 256, 256}}}), layout);
        m2.add_return({rl});
    }

    EXPECT(m1 == m2);
    // reshape_lazy now produces the permuted (NHWC-like) output rather than a standard one
    auto rl1 = std::prev(m1.end())->inputs().front();
    EXPECT(rl1->name() == "reshape_lazy");
    EXPECT(not rl1->get_shape().standard());
    EXPECT(rl1->get_shape().lens() == std::vector<std::size_t>{1, 16, 256, 256});
}

// Symbolic analog of propagate_permutation: the leading batch dimension is a symbol that
// threads through the reshape/transpose/reshape_lazy chain. The pass must propagate the
// permutation just as in the static case, producing the same layout/allocate structure.
TEST_CASE(propagate_permutation_symbolic)
{
    using dd = migraphx::shape::dynamic_dimension;
    using migraphx::sym::lit;

    auto n = migraphx::sym::var("N", {1, 8});
    migraphx::shape in{migraphx::shape::float_type,
                       {dd{n}, dd{lit(1)}, dd{lit(1024)}, dd{lit(1024)}}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", in);
        // 0 copies the symbolic batch dim; the spatial dims split into blocks.
        auto r =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 256, 4, 256, 4}}}), x);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 4, 1, 3}}}), r);
        // post-eliminate_contiguous state: a standardizing gpu::contiguous feeds reshape_lazy
        auto alloc = m1.add_instruction(
            migraphx::make_op("allocate",
                              {{"shape",
                                migraphx::to_value(migraphx::shape{migraphx::shape::float_type,
                                                                   t->get_shape().dyn_dims()})}}));
        auto c = m1.add_instruction(migraphx::make_op("gpu::contiguous"), t, alloc);
        auto rl =
            m1.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {0, 16, 256, 256}}}), c);
        m1.add_return({rl});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", in);
        auto r =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 256, 4, 256, 4}}}), x);
        auto t = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 4, 1, 3}}}), r);
        // layout repacks the transpose into the packed memory order reshape_lazy can alias
        auto l_shape = migraphx::shape::from_permutation(
            migraphx::shape::float_type,
            {dd{n}, dd{lit(4)}, dd{lit(4)}, dd{lit(256)}, dd{lit(256)}},
            {0, 3, 4, 1, 2});
        auto alloc = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(l_shape)}}));
        auto layout = m2.add_instruction(
            make_precompile_op(migraphx::make_op("layout", {{"permutation", {0, 3, 4, 1, 2}}})),
            t,
            alloc);
        auto rl = m2.add_instruction(
            migraphx::make_op("reshape_lazy", {{"dims", {0, 16, 256, 256}}}), layout);
        m2.add_return({rl});
    }

    EXPECT(m1 == m2);
    // reshape_lazy now produces the permuted (NHWC-like) symbolic output rather than a standard one
    auto rl1 = std::prev(m1.end())->inputs().front();
    EXPECT(rl1->name() == "reshape_lazy");
    EXPECT(not rl1->get_shape().standard());
    EXPECT(rl1->get_shape().sym_dims() ==
           std::vector<migraphx::sym::expr>{n, lit(16), lit(256), lit(256)});
}

// When the reshape collapses the non-standard input back to a standard layout there is no
// permutation to propagate, so the pass must leave the graph unchanged.
TEST_CASE(no_permutation_noop)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3, 4}});
        auto t =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
        auto alloc = m1.add_instruction(
            migraphx::make_op("allocate",
                              {{"shape",
                                migraphx::to_value(migraphx::shape{migraphx::shape::float_type,
                                                                   t->get_shape().lens()})}}));
        auto c  = m1.add_instruction(migraphx::make_op("gpu::contiguous"), t, alloc);
        auto rl = m1.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {6, 4}}}), c);
        m1.add_return({rl});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Singleton dimensions make some stride orders ambiguous. The permutation recovered from the
// desired layout can then produce a shape that reshape_lazy cannot alias.
TEST_CASE(ambiguous_singleton_strides_noop)
{
    migraphx::module m1;
    {
        migraphx::shape input_shape{migraphx::shape::float_type, {1, 1, 2}, {1, 2, 1}};
        auto x     = m1.add_parameter("x", input_shape);
        auto alloc = m1.add_instruction(
            migraphx::make_op("allocate",
                              {{"shape",
                                migraphx::to_value(migraphx::shape{migraphx::shape::float_type,
                                                                   input_shape.lens()})}}));
        auto c  = m1.add_instruction(migraphx::make_op("gpu::contiguous"), x, alloc);
        auto rl = m1.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 2}}}), c);
        m1.add_return({rl});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// A replacement can be valid for the matched reshape but make a downstream lazy reshape invalid.
TEST_CASE(downstream_reshape_lazy_noop)
{
    migraphx::module m1;
    {
        migraphx::shape input_shape{migraphx::shape::float_type, {1, 4}, {1, 1}};
        auto x = m1.add_parameter("x", input_shape);
        auto alloc =
            m1.add_instruction(migraphx::make_op(
                "allocate",
                {{"shape",
                  migraphx::to_value(
                      migraphx::shape{migraphx::shape::float_type, input_shape.lens()})}}));
        auto c   = m1.add_instruction(migraphx::make_op("gpu::contiguous"), x, alloc);
        auto rl1 = m1.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 4}}}), c);
        auto rl2 = m1.add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {2, 2}}}), rl1);
        m1.add_return({rl2});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
