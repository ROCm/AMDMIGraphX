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
#include <migraphx/gpu/limit_concat.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <test.hpp>

namespace {

migraphx::operation concat_precompile(std::size_t additional, const migraphx::shape& output)
{
    auto concat = migraphx::make_op("concat", {{"axis", 0}});
    return migraphx::make_op("gpu::precompile_op",
                             {{"op", migraphx::to_value(concat)},
                              {"additional_args", additional},
                              {"output_shape", migraphx::to_value(output)}});
}

migraphx::operation chunk_precompile()
{
    return migraphx::make_op(
        "gpu::precompile_op",
        {{"op", migraphx::to_value(migraphx::make_op("concat", {{"axis", 0}}))}});
}

migraphx::instruction_ref add_alloc(migraphx::module& m, const migraphx::shape& s)
{
    return m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
}

} // namespace

// Five inputs with a limit of two become a tree of concats, each with at most
// two tensor inputs. The original output buffer stays on the root.
TEST_CASE(wide_concat_splits_to_limit)
{
    migraphx::shape one{migraphx::shape::float_type, {1}};
    migraphx::shape two{migraphx::shape::float_type, {2}};
    migraphx::shape four{migraphx::shape::float_type, {4}};
    migraphx::shape five{migraphx::shape::float_type, {5}};

    migraphx::module m1;
    {
        auto a     = m1.add_parameter("a", one);
        auto b     = m1.add_parameter("b", one);
        auto c     = m1.add_parameter("c", one);
        auto d     = m1.add_parameter("d", one);
        auto e     = m1.add_parameter("e", one);
        auto alloc = add_alloc(m1, five);
        auto cat   = m1.add_instruction(concat_precompile(1, five), a, b, c, d, e, alloc);
        m1.add_return({cat});
    }
    migraphx::run_passes(m1, {migraphx::gpu::limit_concat_inputs{2}});
    migraphx::run_passes(m1, {migraphx::gpu::limit_concat_inputs{2}});

    migraphx::module m2;
    {
        auto a      = m2.add_parameter("a", one);
        auto b      = m2.add_parameter("b", one);
        auto c      = m2.add_parameter("c", one);
        auto d      = m2.add_parameter("d", one);
        auto e      = m2.add_parameter("e", one);
        auto alloc  = add_alloc(m2, five);
        auto ab     = add_alloc(m2, two);
        auto cat_ab = m2.add_instruction(chunk_precompile(), a, b, ab);
        auto cd     = add_alloc(m2, two);
        auto cat_cd = m2.add_instruction(chunk_precompile(), c, d, cd);
        auto abcd   = add_alloc(m2, four);
        auto cat_abcd = m2.add_instruction(chunk_precompile(), cat_ab, cat_cd, abcd);
        auto cat      = m2.add_instruction(concat_precompile(1, five), cat_abcd, e, alloc);
        m2.add_return({cat});
    }
    EXPECT(m1 == m2);
}

// Trailing pointwise arguments stay on the root. Only the concat tensors are split.
TEST_CASE(wide_concat_keeps_trailing_args)
{
    migraphx::shape one{migraphx::shape::float_type, {1}};
    migraphx::shape two{migraphx::shape::float_type, {2}};
    migraphx::shape four{migraphx::shape::float_type, {4}};
    migraphx::shape five{migraphx::shape::float_type, {5}};

    migraphx::module m1;
    {
        auto a     = m1.add_parameter("a", one);
        auto b     = m1.add_parameter("b", one);
        auto c     = m1.add_parameter("c", one);
        auto d     = m1.add_parameter("d", one);
        auto e     = m1.add_parameter("e", one);
        auto bias  = m1.add_parameter("bias", five);
        auto alloc = add_alloc(m1, five);
        auto cat =
            m1.add_instruction(concat_precompile(2, five), a, b, c, d, e, bias, alloc);
        m1.add_return({cat});
    }
    migraphx::run_passes(m1, {migraphx::gpu::limit_concat_inputs{2}});

    migraphx::module m2;
    {
        auto a        = m2.add_parameter("a", one);
        auto b        = m2.add_parameter("b", one);
        auto c        = m2.add_parameter("c", one);
        auto d        = m2.add_parameter("d", one);
        auto e        = m2.add_parameter("e", one);
        auto bias     = m2.add_parameter("bias", five);
        auto alloc    = add_alloc(m2, five);
        auto ab       = add_alloc(m2, two);
        auto cat_ab   = m2.add_instruction(chunk_precompile(), a, b, ab);
        auto cd       = add_alloc(m2, two);
        auto cat_cd   = m2.add_instruction(chunk_precompile(), c, d, cd);
        auto abcd     = add_alloc(m2, four);
        auto cat_abcd = m2.add_instruction(chunk_precompile(), cat_ab, cat_cd, abcd);
        auto cat =
            m2.add_instruction(concat_precompile(2, five), cat_abcd, e, bias, alloc);
        m2.add_return({cat});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_within_limit_unchanged)
{
    migraphx::module m1;
    migraphx::shape one{migraphx::shape::float_type, {1}};
    migraphx::shape two{migraphx::shape::float_type, {2}};
    auto a     = m1.add_parameter("a", one);
    auto b     = m1.add_parameter("b", one);
    auto alloc = add_alloc(m1, two);
    auto cat   = m1.add_instruction(concat_precompile(1, two), a, b, alloc);
    m1.add_return({cat});

    migraphx::module m2 = m1;
    migraphx::run_passes(m1, {migraphx::gpu::limit_concat_inputs{256}});
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
