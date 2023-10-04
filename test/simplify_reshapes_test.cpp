/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::simplify_reshapes{}, migraphx::dead_code_elimination{}});
}

inline std::vector<std::vector<std::size_t>> to_lens(const std::vector<migraphx::shape>& shapes)
{
    std::vector<std::vector<std::size_t>> result;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(result), [&](const auto& s) {
        return s.lens();
    });
    return result;
}

migraphx::module make_concat_multibroadcast(const std::vector<size_t>& in_lens,
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
        auto u1 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), l);
        auto t1 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), u1);
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {5, 2, 3}}}), t1);
        m2.add_return({mb});
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
        auto u1 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), l);
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 5}}}), u1);
        m2.add_return({mb});
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

    EXPECT(m1 == m2);
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
    auto ct = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto r2 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 112, 56, 56}}}), ct);
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
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    auto cd = std::distance(m.begin(), new_concat);
    auto new_mb =
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "multibroadcast"; });
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
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    auto cd = std::distance(m.begin(), new_concat);
    auto new_mb =
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "multibroadcast"; });
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
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
    auto cd = std::distance(m.begin(), new_concat);
    auto new_mb =
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "multibroadcast"; });
    auto md = std::distance(m.begin(), new_mb);
    EXPECT(cd == md - 1);
    EXPECT(new_concat->get_operator().to_value()["axis"].to<int>() == 2);
}

TEST_CASE(concat_multibroadcasts4)
{
    // Broadcasted batch dim, axis is broadcasted dim
    std::vector<std::size_t> in_lens     = {3, 4};
    std::vector<std::size_t> mbcast_lens = {2, 3, 4};
    const int axis                       = 0;
    auto m                               = make_concat_multibroadcast(in_lens, mbcast_lens, axis);
    auto m1                              = m;
    run_pass(m);
    EXPECT(m1 == m);
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
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
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
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
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
        std::find_if(m.begin(), m.end(), [](auto ins) { return ins.name() == "concat"; });
    EXPECT(bool{new_concat != m.end()});
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
        auto inx                  = m.add_parameter("X", sx);
        std::vector<int64_t> dims = {1, 1, 2, 1, 2, 1};
        auto rspx = m.add_instruction(migraphx::make_op("reshape", {{"dims", dims}}), inx);
        std::vector<int64_t> mb_dims = {1, 2, 2, 2, 2, 3};
        auto mbx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mb_dims}}), rspx);
        auto std_mb                    = m.add_instruction(migraphx::make_op("contiguous"), mbx);
        std::vector<int64_t> orig_dims = {1, 2, 4, 6};
        auto rmb = m.add_instruction(migraphx::make_op("reshape", {{"dims", orig_dims}}), std_mb);
        auto r   = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), rmb);
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
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 3, 2}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

        auto lrsp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
        auto gr   = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
        auto r    = m.add_instruction(migraphx::make_op("sub"), iny, gr);
        m.add_return({r});

        return m;
    };

    auto m = create_resize_module();
    run_pass(m);
    EXPECT(m == create_resize_module());
}

TEST_CASE(optimize_resize_ind_non_brcst)
{
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
    migraphx::shape sy{migraphx::shape::float_type, {1, 1, 4, 6}};
    auto create_resize_module = [&] {
        migraphx::module m;
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
        std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
        auto li              = m.add_literal(migraphx::literal(si, ind));

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

    auto return_xy = [&](bool cond) {
        migraphx::module m;
        auto x = m.add_parameter("X", s);
        auto y = m.add_parameter("Y", s);
        cond ? m.add_return({x}) : m.add_return({y});
        return m;
    };

    auto m = create_where_module(true);
    run_pass(m);
    EXPECT(m == return_xy(true));

    auto m1 = create_where_module(false);
    run_pass(m1);
    EXPECT(m1 == return_xy(false));
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
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_three_concat_inputs)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", s);
        auto iny = m.add_parameter("Y", s);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny, inx);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {18}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_three_inputs_diff_shapes)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 1, 3, 2}};
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 3, 2}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
        auto data   = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), inx, iny);
        auto data_1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {18}}}), data);
        auto r      = m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data_1, li);
        m.add_return({r});
        return m;
    };

    auto m = create_where_module();
    run_pass(m);
    EXPECT(m == create_where_module());
}

TEST_CASE(where_three_lens_diff)
{
    auto create_where_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 2}};
        migraphx::shape sy{migraphx::shape::float_type, {1, 1, 3, 2}};
        auto inx = m.add_parameter("X", sx);
        auto iny = m.add_parameter("Y", sy);

        migraphx::shape si{migraphx::shape::bool_type, {1, 1, 6}};
        std::vector<char> idata(6, 1);
        auto li     = m.add_literal(migraphx::literal(si, idata));
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
        auto mb_inx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto rsp_iny = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 6}}}), iny);
        auto sum     = m.add_instruction(migraphx::make_op("add"), mb_inx, rsp_iny);
        auto r = m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), sum);
        m.add_return({r});

        return m;
    };

    EXPECT(m1 == create_opt_module());
}

TEST_CASE(reshape_input_non_std)
{
    auto create_module = [] {
        migraphx::module m;
        migraphx::shape sx{migraphx::shape::float_type, {1, 4, 1}};
        migraphx::shape sy{migraphx::shape::float_type, {2, 6, 2, 2}};

        auto inx = m.add_parameter("x", sx);
        auto iny = m.add_parameter("y", sy);
        auto mb_inx =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 6}}}), inx);
        auto std_inx = m.add_instruction(migraphx::make_op("contiguous"), mb_inx);
        auto rsp =
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 6}}}), std_inx);
        auto ty =
            m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), iny);
        auto r = m.add_instruction(migraphx::make_op("add"), rsp, ty);
        m.add_return({r});

        return m;
    };

    auto m1 = create_module();
    run_pass(m1);

    EXPECT(m1 == create_module());
}

TEST_CASE(reshape_cont_nonpw)
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
        auto r = m.add_instruction(migraphx::make_op("convolution"), rsp, iny);
        m.add_return({r});

        return m;
    };

    auto m1 = create_module();
    run_pass(m1);

    EXPECT(m1 == create_module());
}

TEST_CASE(transpose_contiguous_reshape_unary)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 8, 5, 5}});
        auto reshape_ins1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        auto transpose_ins = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), reshape_ins1);
        auto cont_ins = m1.add_instruction(migraphx::make_op("contiguous"), transpose_ins);
        auto reshape_ins2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 10, 10}}}), cont_ins);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), reshape_ins2);
        m1.add_instruction(pass_op{}, relu);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {2, 8, 5, 5}});
        auto reshape_ins1 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 2, 5, 5}}}), x);
        auto transpose_ins = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), reshape_ins1);
        auto relu     = m2.add_instruction(migraphx::make_op("relu"), transpose_ins);
        auto cont_ins = m2.add_instruction(migraphx::make_op("contiguous"), relu);
        auto reshape_ins2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 10, 10}}}), cont_ins);
        m2.add_instruction(pass_op{}, reshape_ins2);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_contiguous_squeeze_unary)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 8, 1, 5}});
        auto transpose_ins =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto cont_ins = m1.add_instruction(migraphx::make_op("contiguous"), transpose_ins);
        auto sq_ins   = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), cont_ins);
        auto rsqrt    = m1.add_instruction(migraphx::make_op("rsqrt"), sq_ins);
        m1.add_instruction(pass_op{}, rsqrt);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {2, 8, 1, 5}});
        auto transpose_ins =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto rsqrt    = m2.add_instruction(migraphx::make_op("rsqrt"), transpose_ins);
        auto cont_ins = m2.add_instruction(migraphx::make_op("contiguous"), rsqrt);
        auto sq_ins   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), cont_ins);
        m2.add_instruction(pass_op{}, sq_ins);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(transpose_contiguous_unsqueeze_unary)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 8, 5, 5}});
        auto transpose_ins =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto cont_ins = m1.add_instruction(migraphx::make_op("contiguous"), transpose_ins);
        auto unsq_ins =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), cont_ins);
        auto round = m1.add_instruction(migraphx::make_op("round"), unsq_ins);
        m1.add_instruction(pass_op{}, round);
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {2, 8, 5, 5}});
        auto transpose_ins =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto round    = m2.add_instruction(migraphx::make_op("round"), transpose_ins);
        auto cont_ins = m2.add_instruction(migraphx::make_op("contiguous"), round);
        auto unsq_ins =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), cont_ins);
        m2.add_instruction(pass_op{}, unsq_ins);
    }
    EXPECT(m1 == m2);
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
        auto y_cont = m1.add_instruction(migraphx::make_op("contiguous"), y_trans);
        auto y_rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 6}}}), y_cont);
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
        m1.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), unsqueezed_args);
    }
    // TODO: This could be simplified to a single transpose after concat
    migraphx::module m2 = m1;
    run_pass(m1);
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
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {12}}}), x);
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
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {3}}}), x);
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
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {3}}}), x);
        auto transpose = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 1, 4}}}), unsqueeze);
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), transpose);
        auto squeeze1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice1);
        auto slice2   = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), transpose);
        auto squeeze2   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice2);
        auto transpose2 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), squeeze2);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), transpose);
        auto squeeze3 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), slice3);
        m2.add_return({squeeze1, transpose2, squeeze3});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
