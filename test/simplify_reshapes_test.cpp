/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m,
                         {
                             migraphx::simplify_reshapes{.enable_op_shape_transform_op = true,
                                                         .enable_gather_rewrite        = true},
                             migraphx::eliminate_common_subexpression{},
                             migraphx::dead_code_elimination{},
                         });
}

inline static std::vector<std::vector<std::size_t>>
to_lens(const std::vector<migraphx::shape>& shapes)
{
    std::vector<std::vector<std::size_t>> result;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(result), [&](const auto& s) {
        return s.lens();
    });
    return result;
}

static migraphx::module make_concat_multibroadcast(const std::vector<size_t>& in_lens,
                                                   const std::vector<size_t>& mbcast_lens,
                                                   const int axis)
{
    migraphx::module m;
    auto s = migraphx::shape{migraphx::shape::float_type, in_lens};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto z = m.add_parameter("z", s);
    auto xm =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mbcast_lens}}), x);
    auto ym =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mbcast_lens}}), y);
    auto zm =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mbcast_lens}}), z);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), xm, ym, zm);
    m.add_return({concat});
    return m;
}

TEST_CASE(broadcast_transpose)
{
    migraphx::module m1;
    {
        auto l = m1.add_parameter("x", {migraphx::shape::float_type, {5}});
        auto mb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 5}}}), l);
        auto t1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), mb);
        m1.add_return({t1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l  = m2.add_parameter("x", {migraphx::shape::float_type, {5}});
        auto b  = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {5, 2, 3}}}), l);
        m2.add_return({b});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(broadcast_transpose_opt)
{
    // extra transpose from transformation will be optimized out
    migraphx::module m1;
    {
        auto l = m1.add_parameter("x", {migraphx::shape::float_type, {5}});
        auto mb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 5}}}), l);
        auto t1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), mb);
        m1.add_return({t1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l  = m2.add_parameter("x", {migraphx::shape::float_type, {5}});
        auto b  = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", {3, 2, 5}}}), l);
        m2.add_return({b});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(broadcast_transpose_scalar)
{
    migraphx::module m1;
    {
        auto l = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), l);
        auto t1 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), mb);
        m1.add_return({t1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2}}}), l);
        m2.add_return({mb});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(broadcast_transpose_scalar_multi_use)
{
    // multibroadcast used more than once
    migraphx::module m1;
    {
        auto l = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), l);
        auto t1 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), mb);
        auto id = m1.add_instruction(migraphx::make_op("identity"), mb);
        m1.add_return({t1, id});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2}}}), l);
        auto mb2 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), l);
        auto id = m2.add_instruction(migraphx::make_op("identity"), mb2);
        m2.add_return({mb, id});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(double_contig)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c1 = mm->add_instruction(migraphx::make_op("contiguous"), t1);
    auto c2 = mm->add_instruction(migraphx::make_op("contiguous"), c1);
    mm->add_return({c2});
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 4);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto t2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), t1);
    mm->add_return({t2});
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(double_transpose_contig)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    auto c1 = mm->add_instruction(migraphx::make_op("contiguous"), t1);
    auto t2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), c1);
    auto c2 = mm->add_instruction(migraphx::make_op("contiguous"), t2);
    mm->add_return({c2});
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    mm->add_return({t1});
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 3);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose_sin_pass)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l  = mm->add_literal(get_2x2());
    auto t1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), t1);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(mm->get_output_shapes().back().standard());
    EXPECT(not mm->get_output_shapes().back().transposed());
    // TODO: Fix this
    // EXPECT(std::distance(mm->begin(), mm->end()) == 1);
    auto result = p.eval({}).back();
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose_sin_pass)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l = mm->add_literal(get_2x2());
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l);
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    run_pass(*mm);
    EXPECT(not mm->get_output_shapes().back().standard());
    EXPECT(mm->get_output_shapes().back().transposed());
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result != get_2x2());
}

TEST_CASE(reshape_transpose)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 112, 56, 56}};
    auto x  = m.add_parameter("x", s);
    auto r1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 28, 56, 56}}}), x);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4}}}), r1);
    auto r2 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 112, 56, 56}}}), t);
    m.add_return({r2});
    EXPECT(m.get_output_shapes().back() == s);
    auto n = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == s);
    EXPECT(std::distance(m.begin(), m.end()) == n);
}

TEST_CASE(transpose_contiguous)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x  = m.add_parameter("x", s);
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    auto c1 = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_return({c1});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n);
}

TEST_CASE(transpose_double_contiguous)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {4, 4}};
    auto x  = m.add_parameter("x", s);
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    auto c1 = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto c2 = m.add_instruction(migraphx::make_op("contiguous"), c1);
    m.add_return({c2});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
    EXPECT(m.has_instruction(t));
}

TEST_CASE(transpose_partial1)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t1);
    m.add_return({t2});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(transpose_partial2)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t1);
    auto t3 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), t2);
    m.add_return({t3});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
}

TEST_CASE(transpose_partial3)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t1);
    auto t3 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), t2);
    auto t4 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), t3);
    m.add_return({t4});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 3);
}

TEST_CASE(nop_transpose1)
{
    migraphx::module m;

    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x = m.add_parameter("x", s);
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), x);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(nop_transpose2)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x  = m.add_parameter("x", s);
    auto t1 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), x);
    auto t2 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), t1);
    auto t3 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), t2);
    auto t4 = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), t3);
    m.add_instruction(pass_op{}, t4);
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 4);
}

TEST_CASE(nop_transpose3)
{
    migraphx::module m;

    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x      = m.add_parameter("x", s);
    auto y      = m.add_parameter("y", s);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), x, y);
    auto t1 =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 3}}}), concat);
    auto t2 =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), t1);
    m.add_return({t2});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(nop_convert)
{
    migraphx::module m;

    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3}};
    auto x = m.add_parameter("x", s);
    auto t = m.add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        x);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back() == out_shape);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(nested_reshape)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4, 5, 6, 7}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto rshp1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 3, 4, 5, 42}}}), x);
        auto rshp2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 12, 5, 42}}}), rshp1);
        auto rshp3 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12, 5, 42}}}), rshp2);
        auto rshp4 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 60, 42}}}), rshp3);
        auto rshp5 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {120, 42}}}), rshp4);
        auto rshp6 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {5040}}}), rshp5);
        m1.add_return({rshp6});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto rshp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {5040}}}), x);
        m2.add_return({rshp});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(nested_reshape_contiguous)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4, 5, 6, 7}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto rshp1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 3, 4, 5, 42}}}), x);
        auto c1 = m1.add_instruction(migraphx::make_op("contiguous"), rshp1);
        auto rshp2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 12, 5, 42}}}), c1);
        auto c2 = m1.add_instruction(migraphx::make_op("contiguous"), rshp2);
        auto rshp3 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12, 5, 42}}}), c2);
        auto c3    = m1.add_instruction(migraphx::make_op("contiguous"), rshp3);
        auto rshp4 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 60, 42}}}), c3);
        auto c4    = m1.add_instruction(migraphx::make_op("contiguous"), rshp4);
        auto rshp5 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {120, 42}}}), c4);
        auto c5    = m1.add_instruction(migraphx::make_op("contiguous"), rshp5);
        auto rshp6 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {5040}}}), c5);
        m1.add_return({rshp6});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto rshp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {5040}}}), x);
        m2.add_return({rshp});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(nested_reshape_squeeze)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto rshp    = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 12}}}), x);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), rshp);
        m1.add_return({squeeze});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto rshp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12}}}), x);
        m2.add_return({rshp});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(nested_squeeze_reshape)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), x);
        auto rshp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12}}}), squeeze);
        m1.add_return({rshp});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto rshp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 12}}}), x);
        m2.add_return({rshp});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(squeeze_unsqueeze_scalar)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), x);
        auto unsqueeze =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), squeeze);
        m1.add_return({unsqueeze});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        m2.add_return({x});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(unsqueeze_broadcast_single)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1}};
    migraphx::module m1;
    {
        auto x         = m1.add_parameter("x", s);
        auto unsqueeze = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 8}}}), unsqueeze);
        m1.add_return({broadcast});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", s);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {1, 8}}}), x);
        m2.add_return({broadcast});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_different_axis_1)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_different_axis_2)
{
    // two slices, one with same axis but other with different
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), x, slice2);
        m1.add_return({slice1, add});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), concat);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({slice1, add});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_in_same_order)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {160}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto y   = m2.add_parameter("y", s);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({add});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_in_reverse_order)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {160}}}), concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto y   = m2.add_parameter("y", s);
        auto add = m2.add_instruction(migraphx::make_op("add"), y, x);
        m2.add_return({add});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_inorder_with_empty_slice)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {160}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {360}}}),
            concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add, slice3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {360}}}),
            concat);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({add, slice3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_uneven_len_1)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto z      = m1.add_parameter("z", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {100}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {100}}, {"ends", {160}}}),
            concat);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto slice4 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {420}}}),
            concat);
        auto slice5 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {420}}, {"ends", {480}}}),
            concat);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), slice1, slice4);
        auto add2 = m1.add_instruction(migraphx::make_op("add"), slice2, slice5);
        auto add3 = m1.add_instruction(migraphx::make_op("add"), slice3, z);
        m1.add_return({add1, add2, add3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto z      = m2.add_parameter("z", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {100}}}), concat);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {100}}, {"ends", {160}}}),
            concat);
        auto slice4 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {420}}}),
            concat);
        auto slice5 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {420}}, {"ends", {480}}}),
            concat);
        auto add1 = m2.add_instruction(migraphx::make_op("add"), slice1, slice4);
        auto add2 = m2.add_instruction(migraphx::make_op("add"), slice2, slice5);
        auto add3 = m2.add_instruction(migraphx::make_op("add"), y, z);
        m2.add_return({add1, add2, add3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_uneven_len_2)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {150}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {150}}, {"ends", {300}}}),
            concat);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {300}}, {"ends", {320}}}),
            concat);
        auto add = m1.add_instruction(migraphx::make_op("add"), slice1, slice2);
        m1.add_return({add, slice3});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_multiple_slice_use)
{
    // multiple use for slice1 and slice3, single use for slice2
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto z      = m1.add_parameter("z", s);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {160}}}), concat);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {320}}, {"ends", {480}}}),
            concat);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), slice1, z);
        auto add2 = m1.add_instruction(migraphx::make_op("add"), slice3, x);
        auto sub1 = m1.add_instruction(migraphx::make_op("sub"), slice1, z);
        auto sub2 = m1.add_instruction(migraphx::make_op("sub"), slice3, x);
        auto add3 = m1.add_instruction(migraphx::make_op("add"), sub1, sub2);
        auto sub3 = m1.add_instruction(migraphx::make_op("sub"), add3, slice2);
        m1.add_return({add1, add2, sub3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto y    = m2.add_parameter("y", s);
        auto z    = m2.add_parameter("z", s);
        auto add1 = m2.add_instruction(migraphx::make_op("add"), x, z);
        auto add2 = m2.add_instruction(migraphx::make_op("add"), z, x);
        auto sub1 = m2.add_instruction(migraphx::make_op("sub"), x, z);
        auto sub2 = m2.add_instruction(migraphx::make_op("sub"), z, x);
        auto add3 = m2.add_instruction(migraphx::make_op("add"), sub1, sub2);
        auto sub3 = m2.add_instruction(migraphx::make_op("sub"), add3, y);
        m2.add_return({add1, add2, sub3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_slice_with_multiple_concat_outs)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 160}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 480}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s1);
        auto y      = m1.add_parameter("y", s1);
        auto z      = m1.add_parameter("z", s1);
        auto w      = m1.add_parameter("w", s2);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {160}}, {"ends", {320}}}),
            concat);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), concat, w);
        auto add2 = m1.add_instruction(migraphx::make_op("add"), slice1, z);
        m1.add_return({add1, add2});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s1);
        auto y      = m2.add_parameter("y", s1);
        auto z      = m2.add_parameter("z", s1);
        auto w      = m2.add_parameter("w", s2);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y, z);
        auto add1   = m2.add_instruction(migraphx::make_op("add"), concat, w);
        auto add2   = m2.add_instruction(migraphx::make_op("add"), y, z);
        m2.add_return({add1, add2});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(concat_multibroadcasts1)
{
    // Broadcasted batch dim, new axis < old axis
    std::vector<std::size_t> in_lens     = {3, 4};
    std::vector<std::size_t> mbcast_lens = {2, 3, 4};
    const int axis                       = 2;
    auto m                               = make_concat_multibroadcast(in_lens, mbcast_lens, axis);
    auto out_shape                       = m.get_output_shapes().back();
    auto n                               = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(new_concat != m.end());
    auto cd = std::distance(m.begin(), new_concat);
    auto new_mb = std::find_if(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "multibroadcast"; });
    auto md = std::distance(m.begin(), new_mb);
    EXPECT(cd == md - 1);
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 1);
}

TEST_CASE(concat_multibroadcasts2)
{
    // Broadcasted middle dim, new axis == old axis
    std::vector<std::size_t> in_lens     = {3, 1, 4};
    std::vector<std::size_t> mbcast_lens = {3, 2, 4};
    const int axis                       = 0;
    auto m                               = make_concat_multibroadcast(in_lens, mbcast_lens, axis);
    auto out_shape                       = m.get_output_shapes().back();
    auto n                               = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(new_concat != m.end());
    auto cd = std::distance(m.begin(), new_concat);
    auto new_mb = std::find_if(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "multibroadcast"; });
    auto md = std::distance(m.begin(), new_mb);
    EXPECT(cd == md - 1);
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 0);
}

TEST_CASE(concat_multibroadcasts3)
{
    // Broadcasted middle dim, new axis == old axis
    std::vector<std::size_t> in_lens     = {3, 1, 4};
    std::vector<std::size_t> mbcast_lens = {3, 2, 4};
    const int axis                       = 2;
    auto m                               = make_concat_multibroadcast(in_lens, mbcast_lens, axis);
    auto out_shape                       = m.get_output_shapes().back();
    auto n                               = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(new_concat != m.end());
    auto cd = std::distance(m.begin(), new_concat);
    auto new_mb = std::find_if(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "multibroadcast"; });
    auto md = std::distance(m.begin(), new_mb);
    EXPECT(cd == md - 1);
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 2);
}

// Broadcasted batch dim, axis is broadcasted dim
// matched by find_concat_multibroadcasts but it skips this case
TEST_CASE(concat_multibroadcasts4)
{
    std::vector<std::size_t> in_lens     = {3, 4};
    std::vector<std::size_t> mbcast_lens = {2, 3, 4};
    const int axis                       = 0;
    auto m                               = make_concat_multibroadcast(in_lens, mbcast_lens, axis);
    auto m1                              = m;
    run_pass(m);
    EXPECT(m1 == m);
}

// Matched by find_concat_multibroadcasts but skipped because dimensions other than concat axis do
// not match
TEST_CASE(concat_multibroadcasts5)
{
    migraphx::module m;
    auto s0 = migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1, 64}};
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 1, 60, 64, 192}};
    auto x  = m.add_parameter("x", s0);
    auto y  = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {1, 12, 60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {1, 12, 60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 4}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

// Matched by find_concat_multibroadcasts but skipped because parameter inputs are not the same
// rank.
TEST_CASE(concat_multibroadcasts6)
{
    migraphx::module m;
    auto s0                           = migraphx::shape{migraphx::shape::float_type, {64}};
    auto s1                           = migraphx::shape{migraphx::shape::float_type, {60, 64, 192}};
    auto x                            = m.add_parameter("x", s0);
    auto y                            = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {12, 60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {12, 60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

// Concat axis moved to 2 because rank(in_dims) < rank(out_dims)
// Matched by find_concat_multibroadcasts but skipped because the dimensions
// other than the concat axis are not the same.
// TODO: has common broadcast axes, so can be simplified by moving multibroadcast up to have a
// smaller concat.
TEST_CASE(concat_multibroadcasts7)
{
    migraphx::module m;
    auto s0                           = migraphx::shape{migraphx::shape::float_type, {1, 1, 64}};
    auto s1                           = migraphx::shape{migraphx::shape::float_type, {60, 64, 192}};
    auto x                            = m.add_parameter("x", s0);
    auto y                            = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {1, 12, 60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {1, 12, 60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 4}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

// Shape of inputs to multibroadcasts do not have the same rank.
// Matched by find_concat_multibroadcasts but skipped.
// TODO: has a common broadcast axis, so can be simplified by moving multibroadcast up to have a
// smaller concat.
TEST_CASE(concat_multibroadcasts8)
{
    migraphx::module m;
    auto s0                           = migraphx::shape{migraphx::shape::float_type, {64, 64}};
    auto s1                           = migraphx::shape{migraphx::shape::float_type, {60, 1, 192}};
    auto x                            = m.add_parameter("x", s0);
    auto y                            = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

// Shape of inputs to multibroadcasts do not have a common broadcast axis.
// Matched by find_concat_multibroadcasts, but skipped because the dimensions other than
// the concat axis are not the same.
TEST_CASE(concat_multibroadcasts9)
{
    migraphx::module m;
    auto s0                           = migraphx::shape{migraphx::shape::float_type, {1, 64, 64}};
    auto s1                           = migraphx::shape{migraphx::shape::float_type, {60, 1, 192}};
    auto x                            = m.add_parameter("x", s0);
    auto y                            = m.add_parameter("y", s1);
    std::vector<std::size_t> mb_lens0 = {60, 64, 64};
    std::vector<std::size_t> mb_lens1 = {60, 64, 192};
    auto mb_x = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens0}}), x);
    auto mb_y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_lens1}}), y);
    auto concat_xy = m.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), mb_x, mb_y);
    m.add_return({concat_xy});
    auto m_original = m;
    run_pass(m);
    EXPECT(m == m_original);
}

TEST_CASE(concat_broadcast1)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {1024, 1024}};
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto y  = m1.add_parameter("y", s);
        auto xb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {8, 1024, 1024}}}), x);
        auto yb = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {8, 1024, 1024}}}), y);
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), xb, yb);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
        auto b      = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {8, 1024, 2048}}}), concat);
        m2.add_return({b});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_transpose1)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x  = m.add_parameter("x", s);
    auto y  = m.add_parameter("y", s);
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), concat);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 3);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(new_concat != m.end());
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 3);
}

TEST_CASE(concat_transpose2)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x  = m.add_parameter("x", s);
    auto y  = m.add_parameter("y", s);
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", -1}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), concat);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(new_concat != m.end());
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 1);
}

TEST_CASE(concat_transpose3)
{
    migraphx::module m;

    auto s  = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x  = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}});
    auto y  = m.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {1, 5, 3, 4}});
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), concat);
    m.add_return({t});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    auto new_concat =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "concat"; });
    EXPECT(new_concat != m.end());
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 1);
}

TEST_CASE(concat_transpose4)
{
    migraphx::module m;
    auto sx = migraphx::shape{migraphx::shape::float_type, {1, 1, 12, 64}};
    auto sy = migraphx::shape{migraphx::shape::float_type, {1, 12, 1, 64}};
    auto x  = m.add_parameter("x", sx);
    auto y  = m.add_parameter("y", sy);
    auto xt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
    auto yt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), y);
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xt, yt);
    auto t =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), concat);
    m.add_return({t});

    migraphx::module m1 = m;
    run_pass(m);

    EXPECT(m1 == m);
}

TEST_CASE(concat_unsqueeze)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {11008, 4096}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s);
        auto y          = m1.add_parameter("y", s);
        auto xunsqueeze = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x);
        auto yunsqueeze = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), y);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), xunsqueeze, yunsqueeze);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto unsqueeze =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 22016, 4096}}}), concat);
        m2.add_return({unsqueeze});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_reshape)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {11008, 32, 128}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto y = m1.add_parameter("y", s);
        auto xreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {11008, 4096}}}), x);
        auto yreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {11008, 4096}}}), y);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), xreshape, yreshape);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto reshape =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {22016, 4096}}}), concat);
        m2.add_return({reshape});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_reshape_change_axis)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 256, 1280}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto y = m1.add_parameter("y", s);
        auto xreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 16, 16, 1280}}}), x);
        auto yreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 16, 16, 1280}}}), y);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), xreshape, yreshape);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}), x, y);
        auto reshape =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 16, 16, 2560}}}), concat);
        m2.add_return({reshape});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(concat_reshape_broadcast)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {11008, 32, 1}};
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto y  = m1.add_parameter("y", s);
        auto xb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {11008, 32, 128}}}), x);
        auto yb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {11008, 32, 128}}}), y);
        auto xreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {11008, 4096}}}), xb);
        auto yreshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {11008, 4096}}}), yb);
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), xreshape, yreshape);
        m1.add_return({concat});
    }
    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", s);
        auto y         = m2.add_parameter("y", s);
        auto concat    = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {22016, 32, 128}}}), concat);
        auto reshape =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {22016, 4096}}}), broadcast);
        m2.add_return({reshape});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(nested_concat)
{
    migraphx::module m;

    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x       = m.add_parameter("x", s);
    auto y       = m.add_parameter("y", s);
    auto concat1 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
    auto concat2 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), y, x);
    auto concat3 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), concat1, concat2);
    m.add_return({concat3});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    EXPECT(std::count_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; }) == 1);
}

TEST_CASE(nested_concat_partial)
{
    migraphx::module m;

    auto s = migraphx::shape{migraphx::shape::float_type, {1, 2, 3, 4}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto l = m.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1, 4, 3, 4}}));
    auto concat1 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x, y);
    auto concat2 = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), y, x);
    auto concat3 =
        m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), concat1, concat2, l);
    m.add_return({concat3});
    auto out_shape = m.get_output_shapes().back();
    auto n         = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().lens() == out_shape.lens());
    EXPECT(std::distance(m.begin(), m.end()) == n - 2);
    EXPECT(std::count_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; }) == 1);
}

TEST_CASE(multibroadcast_simplify)
{
    migraphx::module m;

    std::vector<size_t> s_lens{1, 2, 3, 4};
    auto s = migraphx::shape{migraphx::shape::float_type, s_lens};
    auto x = m.add_parameter("x", s);
    auto y = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s_lens}}), x);
    m.add_instruction(migraphx::make_op("mul"), y, y);
    auto n = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == n - 1);
}

TEST_CASE(multibroadcast_unsqueeze_scalar)
{
    migraphx::module m1;
    {
        auto l   = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb  = m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), l);
        auto t1  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), mb);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), t1, t1);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1}}}), l);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), mb, mb);
        m2.add_return({mul});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(multibroadcast_unsqueeze_cont_scalar)
{
    migraphx::module m1;
    {
        auto l   = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb  = m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), l);
        auto cnt = m1.add_instruction(migraphx::make_op("contiguous"), mb);
        auto t1  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), cnt);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), t1, t1);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1}}}), l);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), mb, mb);
        m2.add_return({mul});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(double_slice1)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {256}}}), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {64}}}), slice1);
        m1.add_return({slice2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {64}}, {"ends", {96}}}), x);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(double_slice2)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {128}}}), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {32}}}), slice1);
        m1.add_return({slice2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::int32_type, {256}});
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {64}}}), x);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(double_slice_multi_axes)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::int32_type, {256, 128}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {32}}, {"ends", {128}}}), x);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {32}}}), slice1);
        m1.add_return({slice2});
    }
    run_pass(m1);

    migraphx::module m2;

    {
        auto x     = m2.add_parameter("x", {migraphx::shape::int32_type, {256, 128}});
        auto slice = m2.add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {0, 1}}, {"starts", {32, 0}}, {"ends", {128, 32}}}),
            x);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(optimize_resize)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {1, 2, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3,
                                3, 3, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0, 0,
                                0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), gr);
        m.add_return({r});

        return m;
    };

    auto m1 = create_resize_module();
    run_pass(m1);

    auto create_optimized_module = [&] {
        migraphx::module m;
        auto inx  = m.add_parameter("X", sx);
        auto rspx = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3, 5}}}), inx);
        auto mbx  = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 2, 2, 2, 3}}}), rspx);
        std::vector<int64_t> orig_dims = {1, 2, 4, 6};
        auto rmb  = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 4, 6}}}), mbx);
        auto rmbb = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 4, 6}}}), rmb);
        auto r = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), rmbb);
        m.add_return({r});

        return m;
    };

    EXPECT(m1 == create_optimized_module());
}

TEST_CASE(optimize_resize_flatten)
{
    migraphx::shape sx{migraphx::shape::float_type, {4}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {48}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3,
                                3, 3, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0, 0,
                                0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto gr = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), inx, li);
        auto r  = m.add_instruction(migraphx::make_op("softmax", {{"axis", 0}}), gr);
        m.add_return({r});

        return m;
    };

    auto m1 = create_resize_module();
    run_pass(m1);

    auto create_optimized_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto rspx =
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 1, 2, 1}}}), inx);
        auto mbx = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 2, 3}}}), rspx);
        std::vector<int64_t> orig_dims = {1, 2, 4, 6};
        auto rmb = m.add_instruction(migraphx::make_op("reshape", {{"dims", {48}}}), mbx);
        auto r   = m.add_instruction(migraphx::make_op("softmax", {{"axis", 0}}), rmb);
        m.add_return({r});

        return m;
    };

    EXPECT(m1 == create_optimized_module());
}

TEST_CASE(optimize_resize_ind_not_apply)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {1, 2, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 2, 2, 2, 3,
                                3, 3, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0, 0,
                                0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), gr);
        m.add_return({r});

        return m;
    };

    auto m1 = create_resize_module();
    run_pass(m1);
    EXPECT(m1 == create_resize_module());
}

TEST_CASE(optimize_resize_rsp_dim_1)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 3, 2}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2}}}), inx);
        auto r    = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_resize_ndims_unequal)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 3, 2}};

    migraphx::module m1;
    {
        auto inx = m1.add_parameter("X", sx);
        auto iny = m1.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 3, 2}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m1.add_literal(migraphx::literal(si, ind));

        auto lrsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m1.add_instruction(migraphx::make_op("sub"), iny, gr);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inx = m2.add_parameter("X", sx);
        auto iny = m2.add_parameter("Y", sy);

        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto rsp2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1, 2, 1}}}), rsp1);
        auto mb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 3}}}), rsp2);
        auto rsp3 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {24}}}), mb);
        auto rsp4 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 4, 3, 2}}}), rsp3);
        auto r = m2.add_instruction(migraphx::make_op("sub"), iny, rsp4);
        m2.add_return({r});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(optimize_resize_ind_non_brcst)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 6}};

    migraphx::module m1;
    {
        auto inx = m1.add_parameter("X", sx);
        auto iny = m1.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m1.add_literal(migraphx::literal(si, ind));

        auto lrsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), inx);
        auto gr   = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m1.add_instruction(migraphx::make_op("sub"), iny, gr);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inx = m2.add_parameter("X", sx);
        auto iny = m2.add_parameter("Y", sy);

        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), inx);
        auto slc  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), rsp1);
        auto rsp2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 1, 2, 1}}}), slc);
        auto mb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 2, 2, 2, 3}}}), rsp2);
        auto rsp_y =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 2, 2, 3}}}), iny);
        auto sub  = m2.add_instruction(migraphx::make_op("sub"), rsp_y, mb);
        auto rsp3 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 4, 6}}}), sub);
        m2.add_return({rsp3});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(optimize_resize_ind_non_const)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 6}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
        auto li   = m.add_parameter("ind", si);
        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("sub"), iny, gr);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_where_true)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
    auto create_where_module = [&](bool cond) {
        migraphx::module m;
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(si.elements(), static_cast<char>(cond));
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto create_expected = [&](bool cond) {
        migraphx::module m;
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        auto data = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto unsq = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4, 5}}}), data);
        auto rsp1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12, 1, 1}}}), unsq);
        auto mb = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {12, 6, 1}}}),
                                    rsp1);
        int64_t start = cond ? 1 : 0;
        int64_t end   = cond ? 2 : 1;
        auto slc      = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {start}}, {"ends", {end}}}), mb);
        auto rsp2 =
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 3, 2, 1}}}), slc);
        auto sq = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {4}}}), rsp2);
        m.add_return({sq});
        return m;
    };

    auto m = create_where_module(true);
    run_pass(m);
    auto expected = create_expected(true);
    EXPECT(m.sort() == expected.sort());

    auto m1 = create_where_module(false);
    run_pass(m1);
    auto expected1 = create_expected(false);
    EXPECT(m1.sort() == expected1.sort());
}

TEST_CASE(where_different_cond_values)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata = {1, 1, 0, 1, 0, 1};
        auto li                 = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_axis_nonzero)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};

    migraphx::module m1;
    {
        auto inx = m1.add_parameter("X", s);
        auto iny = m1.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m1.add_literal(migraphx::literal(si, idata));
        auto data   = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), inx, iny);
        auto data_1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inx = m2.add_parameter("X", s);
        auto iny = m2.add_parameter("Y", s);

        auto data = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), inx, iny);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4}}}), data);
        auto tr   = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 2, 3, 0, 4}}}), unsq);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {12, 1, 1}}}), tr);
        auto mb   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {12, 6, 1}}}), rsp1);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), mb);
        auto rsp2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 3, 2, 1}}}), slc);
        auto sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {4}}}), rsp2);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(where_three_concat_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};

    migraphx::module m1;
    {
        auto inx = m1.add_parameter("X", s);
        auto iny = m1.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m1.add_literal(migraphx::literal(si, idata));
        auto data   = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny, inx);
        auto data_1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {18}}}), data);
        auto r      = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inx = m2.add_parameter("X", s);
        auto iny = m2.add_parameter("Y", s);

        auto data = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny, inx);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4, 5}}}), data);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {18, 1, 1}}}), unsq);
        auto mb   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {18, 6, 1}}}), rsp1);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), mb);
        auto rsp2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 3, 2, 1}}}), slc);
        auto sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {4}}}), rsp2);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(where_three_inputs_diff_shapes)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {2, 1, 3, 2}};

    migraphx::module m1;
    {
        auto inx = m1.add_parameter("X", sx);
        auto iny = m1.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m1.add_literal(migraphx::literal(si, idata));
        auto data   = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {18}}}), data);
        auto r      = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inx = m2.add_parameter("X", sx);
        auto iny = m2.add_parameter("Y", sy);

        auto data = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4, 5}}}), data);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {18, 1, 1}}}), unsq);
        auto mb   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {18, 6, 1}}}), rsp1);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), mb);
        auto rsp2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 3, 2, 1}}}), slc);
        auto sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {4}}}), rsp2);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(where_three_lens_diff)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 3, 2}};

    migraphx::module m1;
    {
        auto inx = m1.add_parameter("X", sx);
        auto iny = m1.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 6}};
        std::vector<char> idata(6, 1);
        auto li     = m1.add_literal(migraphx::literal(si, idata));
        auto data   = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inx = m2.add_parameter("X", sx);
        auto iny = m2.add_parameter("Y", sy);

        auto data = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4, 5}}}), data);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {12, 1, 1}}}), unsq);
        auto mb   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {12, 6, 1}}}), rsp1);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), mb);
        auto unsq2 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), slc);
        auto sq    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), unsq2);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_1d_nd_indices)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {6}});
    migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
    std::vector<int> indices = {0, 1, 2, 3, 4, 5};
    auto li                  = m.add_literal(migraphx::literal(si, indices));
    auto g                   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), x, li);
    m.add_return({g});

    run_pass(m);

    migraphx::module expected;
    auto xe       = expected.add_parameter("x", {migraphx::shape::float_type, {6}});
    auto reshaped = expected.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3}}}), xe);
    expected.add_return({reshaped});

    EXPECT(m == expected);
}

TEST_CASE(gather_axis_slice_broadcast)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 4}});
        migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
        std::vector<int> indices = {1, 1, 1, 2, 2, 2};
        auto li                  = m1.add_literal(migraphx::literal(si, indices));
        auto g = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), x, li);
        m1.add_return({g});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::float_type, {2, 4}});
        auto br = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {2, 4, 3}}}), x);
        auto sliced = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {3}}}), br);
        m2.add_return({sliced});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_constant_single_index)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {3, 4, 5}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {1}};
        auto indices = m1.add_literal(migraphx::literal{si, {2}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    // Verify gather was optimized away
    EXPECT(
        std::none_of(m1.begin(), m1.end(), [](const auto& ins) { return ins.name() == "gather"; }));

    // Verify output shape is correct: {3, 1, 5}
    auto result =
        std::find_if(m1.begin(), m1.end(), [](const auto& ins) { return ins.name() == "@return"; });
    EXPECT(result != m1.end());
    EXPECT(result->inputs().front()->get_shape().lens() == std::vector<std::size_t>{3, 1, 5});

    // Verify only view operations are used (transpose, slice, reshape, squeeze, unsqueeze,
    // broadcast)
    EXPECT(std::all_of(m1.begin(), m1.end(), [](const auto& ins) {
        return ins.name() == "@param" or ins.name() == "@literal" or ins.name() == "@return" or
               ins.name() == "transpose" or ins.name() == "slice" or ins.name() == "reshape" or
               ins.name() == "squeeze" or ins.name() == "unsqueeze" or
               ins.name() == "multibroadcast" or ins.name() == "broadcast";
    }));
}

TEST_CASE(gather_multi_axis_stride)
{
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("X", {migraphx::shape::float_type, {1, 3, 4, 4}});
        auto flatten = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {48}}}), x);

        migraphx::shape indices_shape{migraphx::shape::int32_type, {2, 3, 1, 4}};
        std::vector<int32_t> indices = {0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35,
                                        4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39};
        auto li                      = m1.add_literal(migraphx::literal{indices_shape, indices});
        auto gather                  = m1.add_instruction(migraphx::make_op("gather"), flatten, li);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("X", {migraphx::shape::float_type, {1, 3, 4, 4}});
        auto tr =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1, 3}}}), x);
        auto sq     = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), tr);
        auto sliced = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), sq);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), sliced);
        m2.add_return({unsq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_flatten_multi_axis_stride)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("X", {migraphx::shape::float_type, {48}});

        migraphx::shape indices_shape{migraphx::shape::int32_type, {24}};
        std::vector<int32_t> indices = {0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35,
                                        4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39};
        auto li                      = m1.add_literal(migraphx::literal{indices_shape, indices});
        auto gather                  = m1.add_instruction(migraphx::make_op("gather"), x, li);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("X", {migraphx::shape::float_type, {48}});
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4, 4}}}), x);
        auto tr =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), rsp);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), tr);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {24}}}), slc);
        m2.add_return({rsp2});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_constant_same_indices)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {3, 4, 5}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {3}};
        auto indices = m1.add_literal(migraphx::literal{si, {1, 1, 1}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {3, 4, 5}};
        auto data = m2.add_parameter("data", s);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), data);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 1, 20}}}), unsq);
        auto mb   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 3, 20}}}), rsp1);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), mb);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 3, 4, 5}}}), slc);
        auto sq   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), rsp2);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_constant_same_indices_1d)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {12}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {3}};
        auto indices = m1.add_literal(migraphx::literal{si, {1, 1, 1}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        auto unsqueeze =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), gather);
        m1.add_return({unsqueeze});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {12}};
        auto data = m2.add_parameter("data", s);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), data);
        auto mb   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {12, 3, 1}}}), unsq);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), mb);
        auto sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), slc);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_constant_sequential_indices)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {5, 6}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {3}};
        auto indices = m1.add_literal(migraphx::literal{si, {1, 2, 3}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {5, 6}};
        auto data = m2.add_parameter("data", s);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {30}}}), data);
        auto slc  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {6}}, {"ends", {24}}}), rsp1);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 6}}}), slc);
        m2.add_return({rsp2});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_constant_sequential_indices_1d)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {30}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {3}};
        auto indices = m1.add_literal(migraphx::literal{si, {1, 2, 3}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {30}};
        auto data = m2.add_parameter("data", s);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), data);
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 30}}}), unsq);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {60}}}), mb);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {31}}}), rsp1);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {10, 3}}}), slc1);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), rsp2);
        auto sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slc2);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_constant_stride_indices_1d)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {30}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {3}};
        auto indices = m1.add_literal(migraphx::literal{si, {1, 5, 9}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {30}};
        auto data = m2.add_parameter("data", s);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {24}}}), data);
        auto rsp  = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 4}}}), slc1);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 2}}, {"starts", {0, 1}}, {"ends", {1, 2}}}),
            rsp);
        auto sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 2}}}), slc2);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_constant_stride_divisible_indices_1d)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {30}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {3}};
        auto indices = m1.add_literal(migraphx::literal{si, {0, 5, 10}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {30}};
        auto data = m2.add_parameter("data", s);
        auto rsp  = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 5}}}), data);
        auto slc  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 2}}, {"starts", {0, 0}}, {"ends", {1, 1}}}),
            rsp);
        auto sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 2}}}), slc);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_constant_stride_divisible_indices_window_1d)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {30}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {3}};
        auto indices = m1.add_literal(migraphx::literal{si, {5, 10, 15}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {30}};
        auto data = m2.add_parameter("data", s);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), data);
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 30}}}), unsq);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {60}}}), mb);
        auto slc1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {5}}, {"ends", {35}}}), rsp1);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 5}}}), slc1);
        auto slc2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 2}}, {"starts", {0, 0}}, {"ends", {1, 1}}}),
            rsp2);
        auto sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 2}}}), slc2);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_constant_stride_divisible_both_indices_1d)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {15}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {3}};
        auto indices = m1.add_literal(migraphx::literal{si, {0, 5, 10}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {15}};
        auto data = m2.add_parameter("data", s);
        auto rsp  = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 5}}}), data);
        auto slc  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), rsp);
        auto sq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), slc);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_sequential_stride_rtr_1d)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {8}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {8}};
        auto indices = m1.add_literal(migraphx::literal{si, {0, 4, 1, 5, 2, 6, 3, 7}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s        = migraphx::shape{migraphx::shape::float_type, {8}};
        auto data     = m2.add_parameter("data", s);
        auto reshape1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4}}}), data);
        auto transpose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), reshape1);
        auto reshape2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), transpose);
        m2.add_return({reshape2});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_sequential_stride_rtr_window_1d)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {12}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {8}};
        auto indices = m1.add_literal(migraphx::literal{si, {1, 4, 7, 10, 2, 5, 8, 11}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {12}};
        auto data = m2.add_parameter("data", s);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 3}}}), data);
        auto tr =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), rsp1);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {3}}}), tr);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), slc);
        m2.add_return({rsp2});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_axis0_half_split_concat)
{
    // This pattern is not optimized - gather remains
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {4, 3}});
        migraphx::shape si{migraphx::shape::int32_type, {4}};
        std::vector<int32_t> indices = {2, 3, 0, 1};
        auto li                      = m1.add_literal(migraphx::literal(si, indices));
        auto g = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), x, li);
        m1.add_return({g});
    }
    auto m2 = m1;
    run_pass(m1);

    // Verify output shape is correct: {4, 3}
    auto result =
        std::find_if(m1.begin(), m1.end(), [](const auto& ins) { return ins.name() == "@return"; });
    EXPECT(result != m1.end());
    EXPECT(result->inputs().front()->get_shape().lens() == std::vector<std::size_t>{4, 3});

    EXPECT(m1.sort() == m2.sort());
}

// TEST_CASE(gather_stride_slice)
// {
//     migraphx::module m;
//     auto x            = m.add_parameter("X", {migraphx::shape::float_type, {1, 8}});
//     auto reshape_flat = m.add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), x);
//     migraphx::shape si{migraphx::shape::int32_type, {2, 2}};
//     std::vector<int32_t> indices = {1, 5, 2, 6};
//     auto li                      = m.add_literal(migraphx::literal{si, indices});
//     auto g                       = m.add_instruction(migraphx::make_op("gather"), reshape_flat,
//     li); m.add_return({g});

//     run_pass(m);

//     migraphx::module expected;
//     auto xe = expected.add_parameter("X", {migraphx::shape::float_type, {1, 8}});
//     auto reshape_block =
//         expected.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 4}}}), xe);
//     auto squeeze =
//         expected.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), reshape_block);
//     auto slice = expected.add_instruction(
//         migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {3}}}), squeeze);
//     auto transpose =
//         expected.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}),
//         slice);
//     expected.add_return({transpose});

//     EXPECT(m == expected);
// }

TEST_CASE(gather_flatten_stride_slice)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("X", {migraphx::shape::float_type, {8}});
        migraphx::shape si{migraphx::shape::int32_type, {4}};
        std::vector<int32_t> indices = {1, 5, 2, 6};
        auto li                      = m1.add_literal(migraphx::literal{si, indices});
        auto g                       = m1.add_instruction(migraphx::make_op("gather"), x, li);
        m1.add_return({g});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("X", {migraphx::shape::float_type, {8}});
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4}}}), x);
        auto tr =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), rsp);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {3}}}), tr);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), slc);
        m2.add_return({rsp2});
    }

    EXPECT(m1.sort() == m2.sort());
}

// TEST_CASE(gather_stride_first)
// {
//     migraphx::module m;
//     auto x            = m.add_parameter("X", {migraphx::shape::float_type, {1, 8}});
//     auto reshape_flat = m.add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), x);
//     migraphx::shape si{migraphx::shape::int32_type, {1, 4}};
//     std::vector<int32_t> indices = {0, 2, 4, 6};
//     auto li                      = m.add_literal(migraphx::literal{si, indices});
//     auto g                       = m.add_instruction(migraphx::make_op("gather"), reshape_flat,
//     li); m.add_return({g});

//     run_pass(m);

//     migraphx::module expected;
//     auto xe = expected.add_parameter("X", {migraphx::shape::float_type, {1, 8}});
//     auto reshape_block =
//         expected.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 2}}}), xe);
//     auto squeeze =
//         expected.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), reshape_block);
//     auto slice = expected.add_instruction(
//         migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), squeeze);
//     auto unsqueeze =
//         expected.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), slice);
//     auto result =
//         expected.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), unsqueeze);
//     expected.add_return({result});

//     EXPECT(m == expected);
// }

TEST_CASE(gather_flatten_stride_first)
{
    migraphx::module m;
    auto x = m.add_parameter("X", {migraphx::shape::float_type, {8}});
    migraphx::shape si{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> indices = {0, 2, 4, 6};
    auto li                      = m.add_literal(migraphx::literal{si, indices});
    auto g                       = m.add_instruction(migraphx::make_op("gather"), x, li);
    m.add_return({g});

    run_pass(m);

    migraphx::module expected;
    auto xe = expected.add_parameter("X", {migraphx::shape::float_type, {8}});
    auto reshape_block =
        expected.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 2}}}), xe);
    auto slice = expected.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), reshape_block);
    auto result = expected.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), slice);
    expected.add_return({result});

    EXPECT(m == expected);
}

// TEST_CASE(gather_stride_offset)
// {
//     migraphx::module m;
//     auto x            = m.add_parameter("X", {migraphx::shape::float_type, {1, 16}});
//     auto reshape_flat = m.add_instruction(migraphx::make_op("reshape", {{"dims", {16}}}), x);
//     migraphx::shape si{migraphx::shape::int32_type, {1, 4}};
//     std::vector<int32_t> indices = {1, 5, 9, 13};
//     auto li                      = m.add_literal(migraphx::literal{si, indices});
//     auto g                       = m.add_instruction(migraphx::make_op("gather"), reshape_flat,
//     li); m.add_return({g});

//     run_pass(m);

//     migraphx::module expected;
//     auto xe = expected.add_parameter("X", {migraphx::shape::float_type, {1, 16}});
//     auto reshape_block =
//         expected.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 4}}}), xe);
//     auto squeeze =
//         expected.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), reshape_block);
//     auto slice = expected.add_instruction(
//         migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), squeeze);
//     auto unsqueeze =
//         expected.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), slice);
//     auto result =
//         expected.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), unsqueeze);
//     expected.add_return({result});

//     EXPECT(m == expected);
// }

TEST_CASE(gather_flatten_stride_offset)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("X", {migraphx::shape::float_type, {16}});
        migraphx::shape si{migraphx::shape::int32_type, {1, 4}};
        std::vector<int32_t> indices = {1, 5, 9, 13};
        auto li                      = m1.add_literal(migraphx::literal{si, indices});
        auto g                       = m1.add_instruction(migraphx::make_op("gather"), x, li);
        m1.add_return({g});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("X", {migraphx::shape::float_type, {16}});
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 4}}}), x);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), rsp);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), slc);
        auto sq   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), unsq);
        m2.add_return({sq});
    }

    EXPECT(m1.sort() == m2.sort());
}

// TEST_CASE(gather_stride_grid)
// {
//     migraphx::module m;
//     auto x            = m.add_parameter("X", {migraphx::shape::float_type, {1, 3, 16, 16}});
//     auto reshape_flat = m.add_instruction(migraphx::make_op("reshape", {{"dims", {768}}}), x);
//     migraphx::shape si{migraphx::shape::int32_type, {1, 3, 4, 4}};
//     std::vector<int32_t> indices = {17,  21,  25,  29,  81,  85,  89,  93,  145, 149, 153, 157,
//                                     209, 213, 217, 221, 273, 277, 281, 285, 337, 341, 345, 349,
//                                     401, 405, 409, 413, 465, 469, 473, 477, 529, 533, 537, 541,
//                                     593, 597, 601, 605, 657, 661, 665, 669, 721, 725, 729, 733};
//     auto li                      = m.add_literal(migraphx::literal{si, indices});
//     auto g                       = m.add_instruction(migraphx::make_op("gather"), reshape_flat,
//     li); m.add_return({g});

//     run_pass(m);

//     migraphx::module expected;
//     auto xe = expected.add_parameter("X", {migraphx::shape::float_type, {1, 3, 16, 16}});
//     auto reshape_grid =
//         expected.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 3, 4, 4, 4, 4}}}),
//         xe);
//     auto squeeze_batch =
//         expected.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), reshape_grid);
//     auto slice_inner = expected.add_instruction(
//         migraphx::make_op("slice", {{"axes", {2, 4}}, {"starts", {1, 1}}, {"ends", {2, 2}}}),
//         squeeze_batch);
//     auto unsqueeze_batch =
//         expected.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), slice_inner);
//     auto squeeze_final =
//         expected.add_instruction(migraphx::make_op("squeeze", {{"axes", {3, 5}}}),
//         unsqueeze_batch);
//     expected.add_return({squeeze_final});

//     EXPECT(m == expected);
// }

TEST_CASE(gather_flatten_stride_grid)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("X", {migraphx::shape::float_type, {768}});
        migraphx::shape si{migraphx::shape::int32_type, {48}};
        std::vector<int32_t> indices = {17,  21,  25,  29,  81,  85,  89,  93,  145, 149, 153, 157,
                                        209, 213, 217, 221, 273, 277, 281, 285, 337, 341, 345, 349,
                                        401, 405, 409, 413, 465, 469, 473, 477, 529, 533, 537, 541,
                                        593, 597, 601, 605, 657, 661, 665, 669, 721, 725, 729, 733};
        auto li                      = m1.add_literal(migraphx::literal{si, indices});
        auto g                       = m1.add_instruction(migraphx::make_op("gather"), x, li);
        m1.add_return({g});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("X", {migraphx::shape::float_type, {768}});
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {12, 16, 4}}}), x);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1, 2}}, {"starts", {4, 1}}, {"ends", {8, 2}}}),
            rsp);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {48}}}), slc);
        m2.add_return({rsp2});
    }

    EXPECT(m1.sort() == m2.sort());
}

// TEST_CASE(gather_permutation)
// {
//     migraphx::module m;
//     auto x            = m.add_parameter("X", {migraphx::shape::float_type, {1, 1, 4, 4}});
//     auto reshape_flat = m.add_instruction(migraphx::make_op("reshape", {{"dims", {16}}}), x);
//     migraphx::shape si{migraphx::shape::int32_type, {4, 1, 2, 2}};
//     std::vector<int32_t> indices = {0, 2, 8, 10, 4, 6, 12, 14, 1, 3, 9, 11, 5, 7, 13, 15};
//     auto li                      = m.add_literal(migraphx::literal{si, indices});
//     auto g                       = m.add_instruction(migraphx::make_op("gather"), reshape_flat,
//     li); m.add_return({g});

//     run_pass(m);

//     migraphx::module expected;
//     auto xe = expected.add_parameter("X", {migraphx::shape::float_type, {1, 1, 4, 4}});
//     auto reshape_perm =
//         expected.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 2, 2, 2}}}),
//         xe);
//     auto transpose = expected.add_instruction(
//         migraphx::make_op("transpose", {{"permutation", {5, 3, 0, 1, 2, 4}}}), reshape_perm);
//     auto reshape_out =
//         expected.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 1, 2, 2}}}),
//         transpose);
//     expected.add_return({reshape_out});

//     EXPECT(m == expected);
// }

TEST_CASE(gather_flatten_permutation)
{
    migraphx::module m;
    auto x = m.add_parameter("X", {migraphx::shape::float_type, {16}});
    migraphx::shape si{migraphx::shape::int32_type, {16}};
    std::vector<int32_t> indices = {0, 2, 8, 10, 4, 6, 12, 14, 1, 3, 9, 11, 5, 7, 13, 15};
    auto li                      = m.add_literal(migraphx::literal{si, indices});
    auto g                       = m.add_instruction(migraphx::make_op("gather"), x, li);
    m.add_return({g});

    run_pass(m);

    migraphx::module expected;
    auto xe = expected.add_parameter("X", {migraphx::shape::float_type, {16}});
    auto reshape_perm =
        expected.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2}}}), xe);
    auto transpose = expected.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {3, 1, 0, 2}}}), reshape_perm);
    auto reshape_out =
        expected.add_instruction(migraphx::make_op("reshape", {{"dims", {16}}}), transpose);
    expected.add_return({reshape_out});

    expected.debug_print();

    EXPECT(m == expected);
}

// TEST_CASE(gather_channel_patch)
// {
//     migraphx::module m;
//     auto x            = m.add_parameter("X", {migraphx::shape::float_type, {1, 3, 4, 4}});
//     auto reshape_flat = m.add_instruction(migraphx::make_op("reshape", {{"dims", {48}}}), x);
//     migraphx::shape si{migraphx::shape::int32_type, {4, 3, 1, 1}};
//     std::vector<int32_t> indices = {5, 21, 37, 9, 25, 41, 6, 22, 38, 10, 26, 42};
//     auto li                      = m.add_literal(migraphx::literal{si, indices});
//     auto g                       = m.add_instruction(migraphx::make_op("gather"), reshape_flat,
//     li); m.add_return({g});

//     run_pass(m);

//     migraphx::module expected;
//     auto xe       = expected.add_parameter("X", {migraphx::shape::float_type, {1, 3, 4, 4}});
//     auto slice_hw = expected.add_instruction(
//         migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {1, 1}}, {"ends", {3, 3}}}),
//         xe);
//     auto unsqueeze_hw =
//         expected.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2, 3}}}), slice_hw);
//     auto transpose = expected.add_instruction(
//         migraphx::make_op("transpose", {{"permutation", {5, 4, 0, 1, 2, 3}}}), unsqueeze_hw);
//     auto reshape_out =
//         expected.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 3, 1, 1}}}),
//         transpose);
//     expected.add_return({reshape_out});

//     EXPECT(m == expected);
// }

TEST_CASE(gather_flatten_channel_patch)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("X", {migraphx::shape::float_type, {48}});
        migraphx::shape si{migraphx::shape::int32_type, {12}};
        std::vector<int32_t> indices = {5, 21, 37, 9, 25, 41, 6, 22, 38, 10, 26, 42};
        auto li                      = m1.add_literal(migraphx::literal{si, indices});
        auto g                       = m1.add_instruction(migraphx::make_op("gather"), x, li);
        m1.add_return({g});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("X", {migraphx::shape::float_type, {48}});
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4, 4}}}), x);
        auto tr =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 1, 0}}}), rsp);
        auto slc = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {1, 1}}, {"ends", {3, 3}}}),
            tr);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), slc);
        m2.add_return({rsp2});
    }

    EXPECT(m1.sort() == m2.sort());
}

// TEST_CASE(gather_channel_parity_permutation)
// {
//     migraphx::module m;
//     auto x            = m.add_parameter("X", {migraphx::shape::float_type, {1, 3, 4, 4}});
//     auto reshape_flat = m.add_instruction(migraphx::make_op("reshape", {{"dims", {48}}}), x);
//     migraphx::shape si{migraphx::shape::int32_type, {4, 3, 2, 2}};
//     std::vector<int32_t> indices = {0,  2,  8,  10, 16, 18, 24, 26, 32, 34, 40, 42, 4,  6,  12,
//     14,
//                                     20, 22, 28, 30, 36, 38, 44, 46, 1,  3,  9,  11, 17, 19, 25,
//                                     27, 33, 35, 41, 43, 5,  7,  13, 15, 21, 23, 29, 31, 37, 39,
//                                     45, 47};
//     auto li                      = m.add_literal(migraphx::literal{si, indices});
//     auto g                       = m.add_instruction(migraphx::make_op("gather"), reshape_flat,
//     li); m.add_return({g});

//     run_pass(m);

//     migraphx::module expected;
//     auto xe = expected.add_parameter("X", {migraphx::shape::float_type, {1, 3, 4, 4}});
//     auto reshape_block =
//         expected.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 3, 2, 2, 2, 2}}}),
//         xe);
//     auto transpose = expected.add_instruction(
//         migraphx::make_op("transpose", {{"permutation", {5, 3, 0, 1, 2, 4}}}), reshape_block);
//     auto reshape_out =
//         expected.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 3, 2, 2}}}),
//         transpose);
//     expected.add_return({reshape_out});

//     EXPECT(m == expected);
// }

TEST_CASE(gather_flatten_channel_parity_permutation)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("X", {migraphx::shape::float_type, {48}});
        migraphx::shape si{migraphx::shape::int32_type, {48}};
        std::vector<int32_t> indices = {0, 2, 8,  10, 16, 18, 24, 26, 32, 34, 40, 42,
                                        4, 6, 12, 14, 20, 22, 28, 30, 36, 38, 44, 46,
                                        1, 3, 9,  11, 17, 19, 25, 27, 33, 35, 41, 43,
                                        5, 7, 13, 15, 21, 23, 29, 31, 37, 39, 45, 47};
        auto li                      = m1.add_literal(migraphx::literal{si, indices});
        auto g                       = m1.add_instruction(migraphx::make_op("gather"), x, li);
        m1.add_return({g});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("X", {migraphx::shape::float_type, {48}});
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 2, 2, 2}}}), x);
        auto tr  = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {3, 1, 0, 2}}}), rsp);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {48}}}), tr);
        m2.add_return({rsp2});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_axis1_factorized_grid_const)
{
    migraphx::module m1;
    {
        auto data = m1.add_parameter("data", {migraphx::shape::float_type, {3, 8, 5}});
        migraphx::shape si{migraphx::shape::int32_type, {2, 2, 1}};
        std::vector<int32_t> indices = {1, 3, 5, 7};
        auto li                      = m1.add_literal(migraphx::literal{si, indices});
        auto g = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), data, li);
        m1.add_return({g});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto data = m2.add_parameter("data", {migraphx::shape::float_type, {3, 8, 5}});
        auto rsp1 = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<int64_t>{3, 4, 2, 5}}}), data);
        auto rsp2 = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<int64_t>{12, 10}}}), rsp1);
        auto slc  = m2.add_instruction(migraphx::make_op("slice",
                                                         {{"axes", std::vector<int64_t>{1}},
                                                          {"starts", std::vector<int64_t>{5}},
                                                          {"ends", std::vector<int64_t>{10}}}),
                                      rsp2);
        auto rsp3 = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<int64_t>{3, 2, 2, 1, 5}}}), slc);
        m2.add_return({rsp3});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_axis1_factorized_grid_multi_const)
{
    migraphx::module m1;
    {
        auto data = m1.add_parameter("data", {migraphx::shape::float_type, {2, 27, 4}});
        migraphx::shape si{migraphx::shape::int32_type, {3, 1}};
        std::vector<int32_t> indices = {5, 14, 23};
        auto li                      = m1.add_literal(migraphx::literal{si, indices});
        auto g = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), data, li);
        m1.add_return({g});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto data = m2.add_parameter("data", {migraphx::shape::float_type, {2, 27, 4}});
        auto rsp1 = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<int64_t>{2, 3, 9, 4}}}), data);
        auto rsp2 = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<int64_t>{6, 36}}}), rsp1);
        auto slc  = m2.add_instruction(migraphx::make_op("slice",
                                                         {{"axes", std::vector<int64_t>{1}},
                                                          {"starts", std::vector<int64_t>{20}},
                                                          {"ends", std::vector<int64_t>{24}}}),
                                      rsp2);
        auto rsp3 = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<int64_t>{2, 3, 1, 4}}}), slc);
        m2.add_return({rsp3});
    }

    EXPECT(m1.sort() == m2.sort());
}

// TEST_CASE(gather_constant_scalar_index)
// {
//     migraphx::module m1;
//     {
//         auto s    = migraphx::shape{migraphx::shape::float_type, {3, 4}};
//         auto data = m1.add_parameter("data", s);
//         migraphx::shape si{migraphx::shape::int32_type};
//         auto indices = m1.add_literal(migraphx::literal{si, {2}});
//         auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data,
//         indices); m1.add_return({gather});
//     }
//     run_pass(m1);

//     migraphx::module m2;
//     {
//         auto s     = migraphx::shape{migraphx::shape::float_type, {3, 4}};
//         auto data  = m2.add_parameter("data", s);
//         auto slice = m2.add_instruction(
//             migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), data);
//         auto squeeze = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice);
//         m2.add_return({squeeze});
//     }

//     EXPECT(m1.sort() == m2.sort());
// }

TEST_CASE(gather_constant_negative_index)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {3, 4}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {1}};
        auto indices = m1.add_literal(migraphx::literal{si, {-1}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s     = migraphx::shape{migraphx::shape::float_type, {3, 4}};
        auto data  = m2.add_parameter("data", s);
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), data);
        m2.add_return({slice});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_non_constant_indices)
{
    // Should not be transformed
    migraphx::module m1;
    {
        auto s       = migraphx::shape{migraphx::shape::float_type, {3, 4}};
        auto si      = migraphx::shape{migraphx::shape::int32_type, {2}};
        auto data    = m1.add_parameter("data", s);
        auto indices = m1.add_parameter("indices", si);
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data, indices);
        m1.add_return({gather});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(gather_axis_1)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {2, 5, 3}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {2}};
        auto indices = m1.add_literal(migraphx::literal{si, {0, 1}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), data, indices);
        m1.add_return({gather});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {2, 5, 3}};
        auto data = m2.add_parameter("data", s);
        auto rsp1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 15}}}), data);
        auto slc  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {6}}}), rsp1);
        auto rsp2 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 3}}}), slc);
        m2.add_return({rsp2});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(gather_onnx_axis_one_ex)
{
    migraphx::module m1;
    {
        auto s    = migraphx::shape{migraphx::shape::float_type, {3, 3}};
        auto data = m1.add_parameter("data", s);
        migraphx::shape si{migraphx::shape::int32_type, {2, 1}};
        auto indices = m1.add_literal(migraphx::literal{si, {0, 2}});
        auto gather = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), data, indices);
        m1.add_return({gather});
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_cont)
{
    auto create_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m.add_parameter("x", sx);
        auto iny = m.add_parameter("y", sy);
        auto mb_inx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto r = m.add_instruction(migraphx::make_op("add"), rsp, iny);
        m.add_return({r});

        return m;
    };

    auto m1 = create_module();
    run_pass(m1);

    auto create_opt_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m.add_parameter("x", sx);
        auto iny = m.add_parameter("y", sy);
        auto rsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 1}}}), inx);
        auto mb_inx = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 6}}}), rsp);
        auto r = m.add_instruction(migraphx::make_op("add"), mb_inx, iny);
        m.add_return({r});

        return m;
    };
    auto m2 = create_opt_module();

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_input_non_std)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
    migraphx::shape sy{migraphx::shape::float_type, {2, 6, 2, 2}};
    migraphx::module m1;
    {
        auto inx = m1.add_parameter("x", sx);
        auto iny = m1.add_parameter("y", sy);
        auto mb_inx =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m1.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto ty = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), iny);
        auto r = m1.add_instruction(migraphx::make_op("add"), rsp, ty);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inx = m2.add_parameter("x", sx);
        auto iny = m2.add_parameter("y", sy);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 1}}}), inx);
        auto mb_inx = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 6}}}), rsp);
        auto ty = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), iny);
        auto r = m2.add_instruction(migraphx::make_op("add"), mb_inx, ty);
        m2.add_return({r});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(reshape_cont_nonpw)
{
    migraphx::module m1;
    {
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m1.add_parameter("x", sx);
        auto iny = m1.add_parameter("y", sy);
        auto mb_inx =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m1.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto r = m1.add_instruction(migraphx::make_op("convolution"), rsp, iny);
        m1.add_return({r});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 2, 2, 6}};

        auto inx = m2.add_parameter("x", sx);
        auto iny = m2.add_parameter("y", sy);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 1}}}), inx);
        auto mb_inx = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 6}}}), rsp);
        auto r = m2.add_instruction(migraphx::make_op("convolution"), mb_inx, iny);
        m2.add_return({r});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(reshape_unary_transpose)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        auto relu      = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), relu);
        m1.add_instruction(pass_op{}, transpose);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), x);
        auto reshape_ins =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), relu);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), reshape_ins);
        m2.add_instruction(pass_op{}, transpose);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(reshape_unary_last)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(pointwise_reshape_unary_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 2, 2, 2, 5, 5}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s1);
        auto y   = m1.add_parameter("y", s1);
        auto z   = m1.add_parameter("z", s2);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), mul);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        auto pw   = m1.add_instruction(migraphx::make_op("add"), z, relu);
        m1.add_instruction(pass_op{}, pw);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s1);
        auto z = m2.add_parameter("z", s2);
        auto reshape_x =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        auto reshape_y =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), y);
        auto mul  = m2.add_instruction(migraphx::make_op("mul"), reshape_x, reshape_y);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), mul);
        auto pw   = m2.add_instruction(migraphx::make_op("add"), z, relu);
        m2.add_instruction(pass_op{}, pw);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(pointwise_reshape_unary_pointwise_multi_use)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 32, 10, 64, 64}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 32, 40960}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s1);
        auto z = m1.add_parameter("z", s2);

        auto add = m1.add_instruction(migraphx::make_op("add"), x, y);
        auto reshape1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 32, 40960}}}), add);
        auto convert2 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), reshape1);
        auto div  = m1.add_instruction(migraphx::make_op("div"), convert2, z);
        auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), add);
        auto reshape2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 32, 40960}}}), sqrt);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        auto reshape3 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 32, 40960}}}), relu);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), reshape2, reshape3);
        m1.add_return({div, mul});
    }

    auto output_shapes = m1.get_output_shapes();
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s1);
        auto z = m2.add_parameter("z", s2);

        auto add      = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto convert2 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), add);
        auto zreshape =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), z);
        auto div = m2.add_instruction(migraphx::make_op("div"), convert2, zreshape);
        auto reshape2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 32, 40960}}}), div);
        auto sqrt = m2.add_instruction(migraphx::make_op("sqrt"), add);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        auto mul  = m2.add_instruction(migraphx::make_op("mul"), sqrt, relu);
        auto reshape3 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 32, 40960}}}), mul);
        m2.add_return({reshape2, reshape3});
    }

    EXPECT(m1.get_output_shapes() == output_shapes);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(literal_reshape_unary_transpose_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 2, 5, 2, 5, 2}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s2);
        auto one = m1.add_literal(migraphx::generate_literal(s1));
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), one);
        auto relu      = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), relu);
        auto pw = m1.add_instruction(migraphx::make_op("add"), x, transpose);
        m1.add_instruction(pass_op{}, pw);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s2);
        auto one  = m2.add_literal(migraphx::generate_literal(s1));
        auto reshape_ins =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), one);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), reshape_ins);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), transpose);
        auto pw   = m2.add_instruction(migraphx::make_op("add"), x, relu);
        m2.add_instruction(pass_op{}, pw);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(reshape_unary_transpose_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 2, 5, 2, 5, 2}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        auto relu      = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), relu);
        auto add = m1.add_instruction(migraphx::make_op("add"), transpose, y);
        m1.add_instruction(pass_op{}, add);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto reshape_ins =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), reshape_ins);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), transpose);
        auto add  = m2.add_instruction(migraphx::make_op("add"), relu, y);
        m2.add_instruction(pass_op{}, add);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(pointwise_reshape_unary)
{
    auto s = migraphx::shape{migraphx::shape::float_type, {2, 8, 5, 5}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto add = m1.add_instruction(migraphx::make_op("add"), x, y);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), add);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), reshape_ins);
        m1.add_instruction(pass_op{}, relu);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto y    = m2.add_parameter("y", s);
        auto add  = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        auto reshape_ins =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), relu);
        m2.add_instruction(pass_op{}, reshape_ins);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(pointwise_reshape_layout_convolution)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {640, 320, 1, 1}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s1);
        auto y   = m1.add_parameter("y", s1);
        auto w   = m1.add_parameter("w", s2);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), mul);
        auto layout = m1.add_instruction(
            migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}}), reshape_ins);
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), layout, w);
        m1.add_instruction(pass_op{}, conv);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s1);
        auto y      = m2.add_parameter("y", s1);
        auto w      = m2.add_parameter("w", s2);
        auto mul    = m2.add_instruction(migraphx::make_op("mul"), x, y);
        auto layout = m2.add_instruction(
            migraphx::make_op("layout", {{"permutation", {0, 3, 4, 1, 2}}}), mul);
        auto reshape_ins =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), layout);
        auto conv = m2.add_instruction(migraphx::make_op("convolution"), reshape_ins, w);
        m2.add_instruction(pass_op{}, conv);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(pointwise_transpose_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 64, 4, 4}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 4, 4, 64}};
    migraphx::module m1;
    {
        auto x         = m1.add_parameter("x", s1);
        auto y         = m1.add_parameter("y", s1);
        auto z         = m1.add_parameter("z", s2);
        auto mul       = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), mul);
        auto add  = m1.add_instruction(migraphx::make_op("add"), transpose, z);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s1);
        auto z = m2.add_parameter("z", s2);
        auto transposex =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto transposey =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
        auto mul  = m2.add_instruction(migraphx::make_op("mul"), transposex, transposey);
        auto add  = m2.add_instruction(migraphx::make_op("add"), mul, z);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(pointwise_squeeze_1x1_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 1}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s1);
        auto y       = m1.add_parameter("y", s1);
        auto z       = m1.add_parameter("z", s2);
        auto mul     = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), mul);
        auto add     = m1.add_instruction(migraphx::make_op("add"), squeeze, z);
        auto relu    = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x          = m2.add_parameter("x", s1);
        auto y          = m2.add_parameter("y", s1);
        auto z          = m2.add_parameter("z", s2);
        auto unsqueezez = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), z);
        auto mul        = m2.add_instruction(migraphx::make_op("mul"), x, y);
        auto add        = m2.add_instruction(migraphx::make_op("add"), mul, unsqueezez);
        auto relu       = m2.add_instruction(migraphx::make_op("relu"), add);
        auto squeeze    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), relu);
        m2.add_return({squeeze});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(scalar_pointwise_unsqueeze_1x1_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1}};
    auto s3 = migraphx::shape{migraphx::shape::float_type, {1, 1}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s1);
        auto y       = m1.add_parameter("y", s2);
        auto z       = m1.add_parameter("z", s3);
        auto mul     = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto squeeze = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), mul);
        auto add     = m1.add_instruction(migraphx::make_op("add"), squeeze, z);
        auto relu    = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto z = m2.add_parameter("z", s3);
        auto broadcastx =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 1}}}), x);
        auto unsqueezey = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), y);
        auto mul        = m2.add_instruction(migraphx::make_op("mul"), broadcastx, unsqueezey);
        auto add        = m2.add_instruction(migraphx::make_op("add"), mul, z);
        auto relu       = m2.add_instruction(migraphx::make_op("relu"), add);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(pointwise_transpose_pointwise_used_twice1)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 64, 4, 4}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 4, 4, 64}};
    migraphx::module m1;
    {
        auto x         = m1.add_parameter("x", s1);
        auto y         = m1.add_parameter("y", s1);
        auto z         = m1.add_parameter("z", s2);
        auto mul       = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), mul);
        auto add  = m1.add_instruction(migraphx::make_op("add"), transpose, z);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu, mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s1);
        auto z = m2.add_parameter("z", s2);
        auto transposex =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto transposey =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
        auto mul          = m2.add_instruction(migraphx::make_op("mul"), transposex, transposey);
        auto transposemul = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), mul);
        auto add  = m2.add_instruction(migraphx::make_op("add"), mul, z);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        m2.add_return({relu, transposemul});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(pointwise_transpose_pointwise_used_twice2)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 64, 4, 4}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 4, 4, 64}};
    migraphx::module m1;
    {
        auto x         = m1.add_parameter("x", s1);
        auto y         = m1.add_parameter("y", s1);
        auto z         = m1.add_parameter("z", s2);
        auto mul       = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), mul);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), transpose, z);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add1);
        auto add2 = m1.add_instruction(migraphx::make_op("add"), x, mul);
        m1.add_return({relu, add2});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(pointwise_squeeze_scalar_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1}, {0}};
    migraphx::module m1;
    {
        auto x       = m1.add_parameter("x", s1);
        auto y       = m1.add_parameter("y", s1);
        auto z       = m1.add_parameter("z", s2);
        auto mul     = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), mul);
        auto add     = m1.add_instruction(migraphx::make_op("add"), squeeze, z);
        auto relu    = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    // TODO: Enable a rewrite for this case. For now just check that we dont crash
    // {
    //     auto x        = m2.add_parameter("x", s1);
    //     auto y        = m2.add_parameter("y", s1);
    //     auto z        = m2.add_parameter("z", s2);
    //     auto squeezex = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), x);
    //     auto squeezey = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), y);
    //     auto mul      = m2.add_instruction(migraphx::make_op("mul"), squeezex, squeezey);
    //     auto add      = m2.add_instruction(migraphx::make_op("add"), mul, z);
    //     auto relu     = m2.add_instruction(migraphx::make_op("relu"), add);
    //     m2.add_return({relu});
    // }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(pointwise_unsqueeze_broadcast_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {64, 1, 1}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {64, 3, 7, 7}};
    migraphx::module m1;
    {
        auto x         = m1.add_parameter("x", s1);
        auto y         = m1.add_parameter("y", s1);
        auto z         = m1.add_parameter("z", s2);
        auto mul       = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto unsqueeze = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), mul);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {64, 3, 7, 7}}}), unsqueeze);
        auto add  = m1.add_instruction(migraphx::make_op("add"), broadcast, z);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(split_pointwise_reshape_transpose_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 77, 1536}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1, 77, 768}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s1);
        auto y      = m1.add_parameter("y", s2);
        auto z      = m1.add_parameter("z", s2);
        auto split1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {768}}}), x);
        auto split2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {768}}, {"ends", {1536}}}), x);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), split1, y);
        auto reshape1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 77, 12, 64}}}), add1);
        auto transpose1 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), reshape1);
        auto scale1  = m1.add_literal(0.5f);
        auto scaleb1 = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 77, 64}}}), scale1);
        auto mul1 = m1.add_instruction(migraphx::make_op("mul"), transpose1, scaleb1);

        auto add2 = m1.add_instruction(migraphx::make_op("add"), split2, z);
        auto reshape2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 77, 12, 64}}}), add2);
        auto transpose2 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), reshape2);
        auto scale2  = m1.add_literal(0.6f);
        auto scaleb2 = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 64, 77}}}), scale2);
        auto mul2 = m1.add_instruction(migraphx::make_op("mul"), transpose2, scaleb2);

        auto dot = m1.add_instruction(migraphx::make_op("dot"), mul1, mul2);
        m1.add_return({dot});
    }
    // For now we dont rewrite since it prevents horizontal fusion
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_unsqueeze_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 32, 40960}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 32, 1, 1, 1}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s1);
        auto y          = m1.add_parameter("y", s2);
        auto reduce_sum = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto unsqueeze =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3, 4}}}), reduce_sum);
        auto add  = m1.add_instruction(migraphx::make_op("add"), unsqueeze, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    // TODO:  Enable a rewrite for this case. For now just check that we dont crash
    migraphx::module m2 = m1;
    // {
    //     auto x          = m2.add_parameter("x", s1);
    //     auto y          = m2.add_parameter("y", s2);
    //     auto unsqueeze =
    //         m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3, 4}}}), x);
    //     auto reduce_sum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3,
    //     4}}}), unsqueeze); auto add  = m2.add_instruction(migraphx::make_op("add"), reduce_sum,
    //     y); auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
    //     m2.add_return({relu});
    // }
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_squeeze_pointwise1)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 8, 1024, 1280}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1, 1024, 1280}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s1);
        auto y          = m1.add_parameter("y", s2);
        auto reduce_sum = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto squeeze =
            m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_sum);
        auto add  = m1.add_instruction(migraphx::make_op("add"), squeeze, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x          = m2.add_parameter("x", s1);
        auto y          = m2.add_parameter("y", s2);
        auto reduce_sum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto unsqueeze  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), y);
        auto add        = m2.add_instruction(migraphx::make_op("add"), reduce_sum, unsqueeze);
        auto relu       = m2.add_instruction(migraphx::make_op("relu"), add);
        auto squeeze    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), relu);
        m2.add_return({squeeze});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_squeeze_pointwise2)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 1024, 1024, 1280}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1, 1024, 1280}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s1);
        auto y          = m1.add_parameter("y", s2);
        auto reduce_sum = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto squeeze =
            m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_sum);
        auto add  = m1.add_instruction(migraphx::make_op("add"), squeeze, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x          = m2.add_parameter("x", s1);
        auto y          = m2.add_parameter("y", s2);
        auto reduce_sum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto unsqueeze  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), y);
        auto add        = m2.add_instruction(migraphx::make_op("add"), reduce_sum, unsqueeze);
        auto relu       = m2.add_instruction(migraphx::make_op("relu"), add);
        auto squeeze    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), relu);
        m2.add_return({squeeze});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_squeeze_pointwise3)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 32, 1}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto reduce_sum =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), x);
        auto squeeze =
            m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), reduce_sum);
        auto add  = m1.add_instruction(migraphx::make_op("add"), squeeze, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto reduce_sum =
            m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), x);
        auto unsqueeze = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3, 4}}}), y);
        auto add       = m2.add_instruction(migraphx::make_op("add"), reduce_sum, unsqueeze);
        auto relu      = m2.add_instruction(migraphx::make_op("relu"), add);
        auto squeeze   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), relu);
        m2.add_return({squeeze});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_squeeze_broadcast_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 32, 40960}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto reduce_sum =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), x);
        auto squeeze =
            m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {3, 4}}}), reduce_sum);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), squeeze);
        auto add  = m1.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto reduce_sum =
            m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), x);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), reduce_sum);
        auto reshape1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", s1.lens()}}), y);
        auto add      = m2.add_instruction(migraphx::make_op("add"), broadcast, reshape1);
        auto relu     = m2.add_instruction(migraphx::make_op("relu"), add);
        auto reshape2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), relu);
        m2.add_return({reshape2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_broadcast_reshape_pointwise1)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {64, 4}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {8, 8, 2, 2}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s1);
        auto y          = m1.add_parameter("y", s2);
        auto reduce_sum = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto broadcast  = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), reduce_sum);
        auto reshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), broadcast);
        auto add  = m1.add_instruction(migraphx::make_op("add"), reshape, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto y        = m2.add_parameter("y", s2);
        auto reshapex = m2.add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), x);
        auto reduce_sum =
            m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), reshapex);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), reduce_sum);
        auto add  = m2.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_broadcast_reshape_pointwise2)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {2, 32, 40960}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 320, 64, 64}};
    auto s3 = migraphx::shape{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s1);
        auto y          = m1.add_parameter("y", s2);
        auto reduce_sum = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto broadcast  = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), reduce_sum);
        auto reshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), broadcast);
        auto add  = m1.add_instruction(migraphx::make_op("add"), reshape, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto y        = m2.add_parameter("y", s2);
        auto reshapex = m2.add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        auto reduce_sum =
            m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), reshapex);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), reduce_sum);
        auto reshape1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto add      = m2.add_instruction(migraphx::make_op("add"), broadcast, reshape1);
        auto relu     = m2.add_instruction(migraphx::make_op("relu"), add);
        auto reshape2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), relu);
        m2.add_return({reshape2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_transpose_broadcast_pointwise_diff_size)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 128, 128, 3}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1, 3, 256, 256}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto reduce_sum =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), x);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), reduce_sum);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), transpose);
        auto add  = m1.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto transpose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), x);
        auto reduce_sum =
            m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), transpose);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), reduce_sum);
        auto add  = m2.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_unsqueeze_broadcast_pointwise)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 3, 512, 512}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1, 3, 256, 2, 256, 2}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto reduce_sum =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), x);
        auto unsqueeze =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3, 5}}}), reduce_sum);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), unsqueeze);
        auto add  = m1.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto y        = m2.add_parameter("y", s2);
        auto xreshape = m2.add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), x);
        auto reduce_sum =
            m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4, 5}}}), xreshape);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), reduce_sum);
        auto add  = m2.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_unsqueeze_broadcast_transpose_pointwise1)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 512, 512, 3}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1, 3, 256, 2, 256, 2}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto reduce_sum =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), x);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), reduce_sum);
        auto unsqueeze =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3, 5}}}), transpose);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), unsqueeze);
        auto add  = m1.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto xreshape =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 256, 2, 256, 2, 3}}}), x);
        auto xtranspose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 5, 1, 2, 3, 4}}}), xreshape);
        auto reduce_sum = m2.add_instruction(
            migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4, 5}}}), xtranspose);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), reduce_sum);
        auto add  = m2.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_unsqueeze_broadcast_transpose_pointwise2)
{
    auto s1 = migraphx::shape{migraphx::shape::float_type, {1, 512, 512, 3}};
    auto s2 = migraphx::shape{migraphx::shape::float_type, {1, 3, 256, 2, 256, 2}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto reduce_sum =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), x);
        auto transpose1 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), reduce_sum);
        auto unsqueeze =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3, 5}}}), transpose1);
        auto broadcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), unsqueeze);
        auto add  = m1.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        auto transpose2 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), x);
        auto reshape =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), transpose2);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), relu, reshape);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto xreshape =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 256, 2, 256, 2, 3}}}), x);
        auto xtranspose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 5, 1, 2, 3, 4}}}), xreshape);
        auto reduce_sum = m2.add_instruction(
            migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4, 5}}}), xtranspose);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), reduce_sum);
        auto add  = m2.add_instruction(migraphx::make_op("add"), broadcast, y);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        auto mul  = m2.add_instruction(migraphx::make_op("mul"), relu, xtranspose);
        m2.add_return({mul});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(transpose_contiguous_reshape_binary_packed)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {2, 128, 28, 28}});
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
        auto conv1 = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            x,
            w1); // (2, 256, 28, 28)
        auto w2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 256, 1, 1}}));
        auto conv2 = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            conv1,
            w2); // (2, 512, 14, 14)

        auto conv2_rsp1 = m1.add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 2, 2, 128, 14, 14}}}), conv2);
        auto conv2_trans = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), conv2_rsp1);
        auto conv2_cont = m1.add_instruction(migraphx::make_op("contiguous"), conv2_trans);
        auto conv2_rsp2 = m1.add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 128, 28, 28}}}), conv2_cont);
        auto add_ins = m1.add_instruction(migraphx::make_op("add"), conv2_rsp2, x);
        m1.add_instruction(pass_op{}, add_ins);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::float_type, {2, 128, 28, 28}});
        auto w1 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {256, 128, 1, 1}}));
        auto conv1 = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            x,
            w1); // (2, 256, 28, 28)
        auto w2 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 256, 1, 1}}));
        auto conv2 = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            conv1,
            w2); // (2, 512, 14, 14)

        auto conv2_rsp = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 2, 2, 128, 14, 14}}}), conv2);
        auto conv2_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), conv2_rsp);
        auto x_rsp =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 128, 14, 2, 14, 2}}}), x);
        auto add_ins = m2.add_instruction(migraphx::make_op("add"), conv2_trans, x_rsp);
        auto add_rsp =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 128, 28, 28}}}), add_ins);
        m2.add_instruction(pass_op{}, add_rsp);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_contiguous_reshape_binary_broadcast)
{
    migraphx::module m1;
    {
        migraphx::shape sx{migraphx::shape::float_type, {4}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 6, 2, 2}};

        auto x       = m1.add_parameter("x", sx);
        auto y       = m1.add_parameter("y", sy);
        auto x_brcst = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 4, 6}}}), x);
        auto y_trans =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), y);
        auto y_rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 6}}}), y_trans);
        auto r = m1.add_instruction(migraphx::make_op("add"), y_rsp, x_brcst);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_unsqueeze_concat)
{
    migraphx::module m1;
    {
        auto l0 = m1.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto lt0 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
        auto l1 = m1.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto lt1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l1);
        auto l2 = m1.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto lt2 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l2);
        std::vector<migraphx::instruction_ref> args{lt0, lt1, lt2};
        std::vector<migraphx::instruction_ref> unsqueezed_args;
        int64_t axis = 3;

        std::transform(
            args.begin(),
            args.end(),
            std::back_inserter(unsqueezed_args),
            [&](migraphx::instruction_ref arg) {
                return m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {axis}}}), arg);
            });
        auto concat =
            m1.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), unsqueezed_args);
        m1.add_return({concat});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l0 = m2.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto l1 = m2.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        auto l2 = m2.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 2, 1, 1}});
        std::vector<migraphx::instruction_ref> args{l0, l1, l2};
        std::vector<migraphx::instruction_ref> unsqueezed_args;
        int64_t axis = 1;

        std::transform(
            args.begin(),
            args.end(),
            std::back_inserter(unsqueezed_args),
            [&](migraphx::instruction_ref arg) {
                return m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {axis}}}), arg);
            });
        auto concat =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), unsqueezed_args);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 2}}}), concat);
        m2.add_return({transpose});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {12}}}), x);
        auto transpose1 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice1);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {12}}, {"ends", {24}}}), x);
        auto transpose2 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice2);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {24}}, {"ends", {36}}}), x);
        auto transpose3 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice3);
        m1.add_return({transpose1, transpose2, transpose3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto transpose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}),
            transpose);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {12}}, {"ends", {24}}}),
            transpose);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {24}}, {"ends", {36}}}),
            transpose);
        m2.add_return({slice1, slice2, slice3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_unsqueeze)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {4, 1024, 96, 64}});
        auto transpose1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {8}}}),
            transpose1);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {16}}, {"ends", {24}}}),
            transpose1);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {32}}, {"ends", {40}}}),
            transpose1);
        m1.add_return({slice1, slice2, slice3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {4, 1024, 96, 64}});
        auto unsq =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 1024, 12, 8, 64}}}), x);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 4, 1}}}), unsq);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), transpose);
        auto sq1    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), transpose);
        auto sq2    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice2);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {5}}}), transpose);
        auto sq3 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice3);
        m2.add_return({sq1, sq2, sq3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_diff_perm)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {12}}}), x);
        auto transpose1 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice1);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {12}}, {"ends", {24}}}), x);
        auto transpose2 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), slice2);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {24}}, {"ends", {36}}}), x);
        auto transpose3 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice3);
        m1.add_return({transpose1, transpose2, transpose3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto transpose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}),
            transpose);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {12}}, {"ends", {24}}}),
            transpose);
        auto transpose2 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), slice2);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {24}}, {"ends", {36}}}),
            transpose);
        m2.add_return({slice1, transpose2, slice3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_single_transpose)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::float_type, {1, 384, 36, 64}});
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {12}}}), x);
        auto sqrt1  = m1.add_instruction(migraphx::make_op("sqrt"), slice1);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {12}}, {"ends", {24}}}), x);
        auto transpose = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), slice2);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {24}}, {"ends", {36}}}), x);
        auto sqrt3 = m1.add_instruction(migraphx::make_op("sqrt"), slice3);
        m1.add_return({sqrt1, transpose, sqrt3});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_non_packed_axis)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 384, 36, 64}});
        auto transpose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto slice = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}),
            transpose);
        auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), slice);
        m1.add_return({sqrt});
    }
    auto output_shapes = m1.get_output_shapes();
    run_pass(m1);
    EXPECT(m1.get_output_shapes() == output_shapes);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {2, 384, 36, 64}});
        auto unsqueeze =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 384, 3, 12, 64}}}), x);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 1, 4}}}), unsqueeze);
        auto slice = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), transpose);
        auto squeeze = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice);
        auto sqrt    = m2.add_instruction(migraphx::make_op("sqrt"), squeeze);
        m2.add_return({sqrt});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_slice_non_packed_multi_axis)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 384, 36, 64}});
        auto transpose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto slice1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {12}}}),
            transpose);
        auto slice2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {12}}, {"ends", {24}}}),
            transpose);
        auto transpose2 = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), slice2);
        auto slice3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {24}}, {"ends", {36}}}),
            transpose);
        m1.add_return({slice1, transpose2, slice3});
    }
    auto output_shapes = m1.get_output_shapes();
    run_pass(m1);
    EXPECT(to_lens(m1.get_output_shapes()) == to_lens(output_shapes));
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {2, 384, 36, 64}});
        auto unsqueeze =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 384, 3, 12, 64}}}), x);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 1, 4}}}), unsqueeze);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), transpose);
        auto squeeze1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice1);
        auto slice2   = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), transpose);
        auto transpose2 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}), slice2);
        auto squeeze2 =
            m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), transpose2);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), transpose);
        auto squeeze3 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice3);
        m2.add_return({squeeze1, squeeze2, squeeze3});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_reshape_dot)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 10, 32, 16}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 10, 16, 32}};
    migraphx::module m1;
    {
        auto a     = m1.add_literal(migraphx::generate_literal(as));
        auto b     = m1.add_parameter("input", bs);
        auto a_rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {20, 32, 16}}}), a);
        auto b_rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {20, 16, 32}}}), b);

        auto dot = m1.add_instruction(migraphx::make_op("dot"), a_rsp, b_rsp);
        auto dot_rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 10, 32, 32}}}), dot);
        m1.add_return({dot_rsp});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto a   = m2.add_literal(migraphx::generate_literal(as));
        auto b   = m2.add_parameter("input", bs);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a, b);
        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_reshape_dot_gemm_axis)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 10, 512}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 10, 512}};
    migraphx::module m1;
    {
        auto a     = m1.add_literal(migraphx::generate_literal(as));
        auto b     = m1.add_parameter("input", bs);
        auto a_rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {20, 32, 16}}}), a);
        auto b_rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {20, 16, 32}}}), b);

        auto dot = m1.add_instruction(migraphx::make_op("dot"), a_rsp, b_rsp);
        auto dot_rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 10, 1024}}}), dot);
        m1.add_return({dot_rsp});
    };
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {32, 32}};

    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", s_inp);
        auto rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), inp);
        auto w   = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 32}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), rsp, w_bc);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("inp", s_inp);
        auto w    = m2.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 8, 32, 32}}}), w);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), inp, w_bc);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), dot);
        m2.add_return({rsp});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_flipped)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {16, 8}};

    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", s_inp);
        auto rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {16, 8, 32}}}), inp);
        auto w   = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {16, 16, 8}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), w_bc, rsp);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("inp", s_inp);
        auto w    = m2.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 8, 16, 8}}}), w);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), w_bc, inp);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {16, 16, 32}}}), dot);
        m2.add_return({rsp});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_dot_axis)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 4}};
    migraphx::shape s_w{migraphx::shape::float_type, {32, 32}};

    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", s_inp);
        auto rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 8, 32}}}), inp);
        auto w   = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 32}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), rsp, w_bc);
        m1.add_return({dot});
    };

    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_flipped_dot_axis)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {8, 64}};

    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", s_inp);
        auto rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), inp);
        auto w   = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8, 64}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), w_bc, rsp);
        m1.add_return({dot});
    };

    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_broadcast)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {32}};

    migraphx::module m1;
    {
        auto inp  = m1.add_parameter("inp", s_inp);
        auto rsp  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), inp);
        auto w    = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 32}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), rsp, w_bc);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("inp", s_inp);
        auto w    = m2.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", {2, 8, 32, 32}}}), w);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), inp, w_bc);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 64, 32}}}), dot);
        m2.add_return({rsp});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reshape_dot_broadcast_2)
{
    migraphx::shape s_inp{migraphx::shape::float_type, {2, 8, 8, 32}};
    migraphx::shape s_w{migraphx::shape::float_type, {32}};

    migraphx::module m1;
    {
        auto inp  = m1.add_parameter("inp", s_inp);
        auto rsp  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {128, 32}}}), inp);
        auto w    = m1.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {32, 32}}}), w);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), rsp, w_bc);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("inp", s_inp);
        auto w    = m2.add_literal(migraphx::generate_literal(s_w));
        auto w_bc = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 3}, {"out_lens", {2, 8, 32, 32}}}), w);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), inp, w_bc);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {128, 32}}}), dot);
        m2.add_return({rsp});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(mul_transpose)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 32, 64, 64}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 64, 32, 32}};
    migraphx::module m1;
    {
        auto inp   = m1.add_parameter("input", s);
        auto c1    = m1.add_literal(migraphx::generate_literal(s));
        auto mul   = m1.add_instruction(migraphx::make_op("mul"), inp, c1);
        auto trans = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), mul);

        auto c3  = m1.add_literal(migraphx::generate_literal(s2));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), trans, c3);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp       = m2.add_parameter("input", s);
        auto inp_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), inp);
        auto c1 = m2.add_literal(migraphx::generate_literal(s));
        auto c1_trans =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), c1);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), inp_trans, c1_trans);
        auto c3  = m2.add_literal(migraphx::generate_literal(s2));
        auto dot = m2.add_instruction(migraphx::make_op("dot"), mul, c3);
        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(add_transpose)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 32, 64, 64}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 64, 32, 32}};
    migraphx::module m1;
    {
        auto inp   = m1.add_parameter("input", s);
        auto c1    = m1.add_literal(migraphx::generate_literal(s));
        auto mul   = m1.add_instruction(migraphx::make_op("add"), inp, c1);
        auto trans = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), mul);

        auto c3  = m1.add_literal(migraphx::generate_literal(s2));
        auto dot = m1.add_instruction(migraphx::make_op("dot"), trans, c3);
        m1.add_return({dot});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp       = m2.add_parameter("input", s);
        auto inp_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), inp);
        auto c1 = m2.add_literal(migraphx::generate_literal(s));
        auto c1_trans =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), c1);
        auto mul = m2.add_instruction(migraphx::make_op("add"), inp_trans, c1_trans);
        auto c3  = m2.add_literal(migraphx::generate_literal(s2));
        auto dot = m2.add_instruction(migraphx::make_op("dot"), mul, c3);
        m2.add_return({dot});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(flatten)
{
    migraphx::shape s{migraphx::shape::float_type, {4608, 8, 2}};

    migraphx::module m1;
    {
        auto inp  = m1.add_parameter("input", s);
        auto flat = m1.add_instruction(migraphx::make_op("flatten", {{"axis", 1}}), inp);
        m1.add_return({flat});
    };
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp  = m2.add_parameter("input", s);
        auto flat = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4608, 16}}}), inp);
        m2.add_return({flat});
    };

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(conv_add_layernorm_conv)
{
    migraphx::module m1;
    {
        auto p_x =
            m1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 4, 64, 64}});
        auto p_w1 =
            m1.add_parameter("w1", migraphx::shape{migraphx::shape::float_type, {320, 4, 3, 3}});
        auto p_w2 =
            m1.add_parameter("w2", migraphx::shape{migraphx::shape::float_type, {4, 320, 3, 3}});
        auto p_y0 = m1.add_parameter("y0", migraphx::shape{migraphx::shape::float_type, {320}});
        auto p_scale =
            m1.add_parameter("scale", migraphx::shape{migraphx::shape::float_type, {40960}});
        auto p_bias =
            m1.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {40960}});
        auto p_y1  = m1.add_parameter("y1", migraphx::shape{migraphx::shape::float_type, {1}});
        auto p_y2  = m1.add_parameter("y2", migraphx::shape{migraphx::shape::float_type, {1}});
        auto p_y3  = m1.add_parameter("y3", migraphx::shape{migraphx::shape::float_type, {1}});
        auto conv1 = m1.add_instruction(migraphx::make_op("convolution",
                                                          {{"dilation", {1, 1}},
                                                           {"group", 1},
                                                           {"padding", {1, 1, 1, 1}},
                                                           {"padding_mode", 0},
                                                           {"stride", {1, 1}}}),
                                        p_x,
                                        p_w1);
        auto p_y0b = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 320, 64, 64}}}), p_y0);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), conv1, p_y0b);
        auto reshape1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 32, -1}}}), add1);
        auto p_y2b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 40960}}}), p_y2);
        auto div1 = m1.add_instruction(migraphx::make_op("div"), reshape1, p_y2b);
        auto reduce_sum1 =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), div1);
        auto reduce_sum1b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 40960}}}), reduce_sum1);
        auto sub1  = m1.add_instruction(migraphx::make_op("sub"), reshape1, reduce_sum1b);
        auto mul1  = m1.add_instruction(migraphx::make_op("mul"), reshape1, reshape1);
        auto p_y3b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 40960}}}), p_y3);
        auto div2 = m1.add_instruction(migraphx::make_op("div"), mul1, p_y3b);
        auto reduce_sum2 =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), div2);
        auto mul2  = m1.add_instruction(migraphx::make_op("mul"), reduce_sum1, reduce_sum1);
        auto sub2  = m1.add_instruction(migraphx::make_op("sub"), reduce_sum2, mul2);
        auto p_y1b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 1}}}), p_y1);
        auto add2  = m1.add_instruction(migraphx::make_op("add"), sub2, p_y1b);
        auto sqrt  = m1.add_instruction(migraphx::make_op("sqrt"), add2);
        auto sqrtb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 40960}}}), sqrt);
        auto div3     = m1.add_instruction(migraphx::make_op("div"), sub1, sqrtb);
        auto p_scaleb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 40960}}}), p_scale);
        auto mul3    = m1.add_instruction(migraphx::make_op("mul"), div3, p_scaleb);
        auto p_biasb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 40960}}}), p_bias);
        auto add3 = m1.add_instruction(migraphx::make_op("add"), mul3, p_biasb);
        auto reshape2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {0, 320, 64, 64}}}), add3);
        auto conv2 = m1.add_instruction(migraphx::make_op("convolution",
                                                          {{"dilation", {1, 1}},
                                                           {"group", 1},
                                                           {"padding", {1, 1, 1, 1}},
                                                           {"padding_mode", 0},
                                                           {"stride", {1, 1}}}),
                                        reshape2,
                                        p_w2);
        m1.add_return({conv2});
    };
    run_pass(m1);
    migraphx::module m2;
    {
        auto p_y3 = m2.add_parameter("y3", migraphx::shape{migraphx::shape::float_type, {1}});
        auto p_y2 = m2.add_parameter("y2", migraphx::shape{migraphx::shape::float_type, {1}});
        auto p_y1 = m2.add_parameter("y1", migraphx::shape{migraphx::shape::float_type, {1}});
        auto p_bias =
            m2.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {40960}});
        auto p_scale =
            m2.add_parameter("scale", migraphx::shape{migraphx::shape::float_type, {40960}});
        auto p_y0 = m2.add_parameter("y0", migraphx::shape{migraphx::shape::float_type, {320}});
        auto p_w2 =
            m2.add_parameter("w2", migraphx::shape{migraphx::shape::float_type, {4, 320, 3, 3}});
        auto p_w1 =
            m2.add_parameter("w1", migraphx::shape{migraphx::shape::float_type, {320, 4, 3, 3}});
        auto p_x =
            m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 4, 64, 64}});
        auto conv1    = m2.add_instruction(migraphx::make_op("convolution",
                                                             {{"dilation", {1, 1}},
                                                              {"group", 1},
                                                              {"padding", {1, 1, 1, 1}},
                                                              {"padding_mode", 0},
                                                              {"stride", {1, 1}}}),
                                        p_x,
                                        p_w1);
        auto reshape1 = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv1);
        auto reshape2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {32, 10}}}), p_y0);
        auto reshape2b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}),
            reshape2);
        auto add1           = m2.add_instruction(migraphx::make_op("add"), reshape1, reshape2b);
        auto unsqueeze_p_y2 = m2.add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3, 4}}, {"steps", {}}}), p_y2);
        auto unsqueeze_p_y2b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 10, 64, 64}}}),
            unsqueeze_p_y2);
        auto div1 = m2.add_instruction(migraphx::make_op("div"), add1, unsqueeze_p_y2b);
        auto reduce_sum1 =
            m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), div1);
        auto reduce_sum1b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 10, 64, 64}}}), reduce_sum1);
        auto sub1           = m2.add_instruction(migraphx::make_op("sub"), add1, reduce_sum1b);
        auto mul1           = m2.add_instruction(migraphx::make_op("mul"), add1, add1);
        auto unsqueeze_p_y3 = m2.add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3, 4}}, {"steps", {}}}), p_y3);
        auto p_y3b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 10, 64, 64}}}),
            unsqueeze_p_y3);
        auto div2 = m2.add_instruction(migraphx::make_op("div"), mul1, p_y3b);
        auto reduce_sum2 =
            m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), div2);
        auto mul2 = m2.add_instruction(migraphx::make_op("mul"), reduce_sum1, reduce_sum1);
        auto sub2 = m2.add_instruction(migraphx::make_op("sub"), reduce_sum2, mul2);
        auto unsqueeze_p_y1 = m2.add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", {1, 2}}, {"steps", {}}}), p_y1);
        auto unsqueeze_p_y1b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", {2, 32, 1, 1, 1}}}),
            unsqueeze_p_y1);
        auto add2  = m2.add_instruction(migraphx::make_op("add"), sub2, unsqueeze_p_y1b);
        auto sqrt  = m2.add_instruction(migraphx::make_op("sqrt"), add2);
        auto sqrtb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 32, 10, 64, 64}}}), sqrt);
        auto div3 = m2.add_instruction(migraphx::make_op("div"), sub1, sqrtb);
        auto reshape6 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {10, 64, 64}}}), p_scale);
        auto reshape6b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", {2, 32, 10, 64, 64}}}),
            reshape6);
        auto mul3 = m2.add_instruction(migraphx::make_op("mul"), div3, reshape6b);
        auto reshape7 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {10, 64, 64}}}), p_bias);
        auto reshape7b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", {2, 32, 10, 64, 64}}}),
            reshape7);
        auto add3 = m2.add_instruction(migraphx::make_op("add"), mul3, reshape7b);
        auto reshape8 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), add3);
        auto conv2 = m2.add_instruction(migraphx::make_op("convolution",
                                                          {{"dilation", {1, 1}},
                                                           {"group", 1},
                                                           {"padding", {1, 1, 1, 1}},
                                                           {"padding_mode", 0},
                                                           {"stride", {1, 1}}}),
                                        reshape8,
                                        p_w2);
        m2.add_return({conv2});
    };

    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
