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
#include <migraphx/normalize_ops.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

using dd = migraphx::shape::dynamic_dimension;
using migraphx::sym::lit;
using migraphx::sym::var;

struct normalize_test_op
{
    std::vector<int64_t> axes = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.axes, "axes"));
    }

    migraphx::value attributes() const
    {
        migraphx::value normalize;
        normalize["axes"] = migraphx::value::array{migraphx::op::normalize_attribute::clip_max,
                                                   migraphx::op::normalize_attribute::clip_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "normalize_ops_test::test_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return migraphx::argument{output_shape};
    }
};

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::normalize_ops{}, migraphx::dead_code_elimination{}});
}

static migraphx::module create_gather(int64_t axis)
{
    migraphx::module m;
    migraphx::shape sd{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape si{migraphx::shape::int64_type, {2, 3}};
    auto di = m.add_parameter("data", sd);
    auto ii = m.add_parameter("ind", si);
    auto r  = m.add_instruction(migraphx::make_op("gather", {{"axis", axis}}), di, ii);
    m.add_return({r});

    return m;
}

TEST_CASE(gather_test)
{

    auto m1 = create_gather(-3);
    auto m2 = create_gather(0);
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(gather_test_1)
{
    auto m1 = create_gather(1);
    auto m2 = create_gather(1);
    run_pass(m1);

    EXPECT(m1 == m2);
}

static migraphx::module create_padded_op(const std::vector<size_t>& pad_vals)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    auto si = m.add_parameter("data", s);
    auto r  = m.add_instruction(migraphx::make_op("pooling", {{"padding", pad_vals}}), si);
    m.add_return({r});

    return m;
}

TEST_CASE(padding_attr_test)
{
    migraphx::module m1 = create_padded_op({0, 1});
    migraphx::module m2 = create_padded_op({0, 1, 0, 1});
    run_pass(m1);

    EXPECT(m1 == m2);
}

static migraphx::module create_reduce_mean(const std::vector<int64_t>& axes)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    auto si = m.add_parameter("data", s);
    auto r  = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), si);
    m.add_return({r});

    return m;
}

TEST_CASE(reduce_mean_test)
{
    migraphx::module m1 = create_reduce_mean({0, 1, -1});
    migraphx::module m2 = create_reduce_mean({0, 1, 3});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(reduce_mean_test_1)
{
    migraphx::module m1 = create_reduce_mean({0, 1, 2});
    migraphx::module m2 = create_reduce_mean({0, 1, 2});
    run_pass(m1);

    EXPECT(m1 == m2);
}

static migraphx::module create_slice(const std::vector<int64_t>& axes,
                                     const std::vector<int64_t>& starts,
                                     const std::vector<int64_t>& ends)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    auto si = m.add_parameter("data", s);
    auto r  = m.add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), si);
    m.add_return({r});

    return m;
}

TEST_CASE(slice_test)
{
    migraphx::module m1 = create_slice({0, 1, -1}, {-5, 1, -3}, {2, 2, 8});
    migraphx::module m2 = create_slice({0, 1, 3}, {0, 1, 2}, {2, 2, 5});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(slice_test_1)
{
    migraphx::module m1 = create_slice({0, 1, 3}, {0, 1, -3}, {1, 2, 5});
    migraphx::module m2 = create_slice({0, 1, 3}, {0, 1, 2}, {1, 2, 5});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(slice_sym_ends_clamped_test)
{
    // n is not provably ordered against the axis length 5, so the bound clamps to min(n, 5).
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    migraphx::shape sb{migraphx::shape::int64_type, {1}};

    migraphx::module m1;
    {
        auto data = m1.add_parameter("data", s);
        auto ends = m1.add_parameter("ends", sb);
        auto r    = m1.add_instruction(
            migraphx::make_op("slice",
                                 {{"axes", {3}},
                                  {"starts", {0}},
                                  {"ends", migraphx::value::array{migraphx::to_value(dd{n})}},
                                  {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m1.add_return({r});
    }

    migraphx::module m2;
    {
        auto data = m2.add_parameter("data", s);
        auto ends = m2.add_parameter("ends", sb);
        auto r    = m2.add_instruction(
            migraphx::make_op(
                "slice",
                {{"axes", {3}},
                    {"starts", {0}},
                    {"ends",
                     migraphx::value::array{migraphx::to_value(dd{migraphx::sym::min(n, lit(5))})}},
                    {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m2.add_return({r});
    }
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(slice_sym_ends_below_len_test)
{
    // n < 5 is provable, so the bound keeps the bare symbol rather than gaining a min wrapper.
    auto n = var("n", {1, 4});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    migraphx::shape sb{migraphx::shape::int64_type, {1}};

    migraphx::module m1;
    {
        auto data = m1.add_parameter("data", s);
        auto ends = m1.add_parameter("ends", sb);
        auto r    = m1.add_instruction(
            migraphx::make_op("slice",
                                 {{"axes", {3}},
                                  {"starts", {0}},
                                  {"ends", migraphx::value::array{migraphx::to_value(dd{n})}},
                                  {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m1.add_return({r});
    }

    migraphx::module m2;
    {
        auto data = m2.add_parameter("data", s);
        auto ends = m2.add_parameter("ends", sb);
        auto r    = m2.add_instruction(
            migraphx::make_op("slice",
                                 {{"axes", {3}},
                                  {"starts", {0}},
                                  {"ends", migraphx::value::array{migraphx::to_value(dd{n})}},
                                  {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m2.add_return({r});
    }
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(slice_sym_ends_at_len_test)
{
    // n >= 5 is provable, so the bound collapses to the axis length and demotes to an integer.
    auto n = var("n", {6, 9});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    migraphx::shape sb{migraphx::shape::int64_type, {1}};

    migraphx::module m1;
    {
        auto data = m1.add_parameter("data", s);
        auto ends = m1.add_parameter("ends", sb);
        auto r    = m1.add_instruction(
            migraphx::make_op("slice",
                                 {{"axes", {3}},
                                  {"starts", {0}},
                                  {"ends", migraphx::value::array{migraphx::to_value(dd{n})}},
                                  {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m1.add_return({r});
    }

    migraphx::module m2;
    {
        auto data = m2.add_parameter("data", s);
        auto ends = m2.add_parameter("ends", sb);
        auto r    = m2.add_instruction(migraphx::make_op("slice",
                                                         {{"axes", {3}},
                                                          {"starts", {0}},
                                                          {"ends", {5}},
                                                          {"mode", migraphx::value::array{"ends"}}}),
                                    data,
                                    ends);
        m2.add_return({r});
    }
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(slice_sym_starts_clamped_test)
{
    // The starts attribute normalizes the same way. Axis 1 has length 3.
    auto n = var("n", {0, 5});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    migraphx::shape sb{migraphx::shape::int64_type, {1}};

    migraphx::module m1;
    {
        auto data   = m1.add_parameter("data", s);
        auto starts = m1.add_parameter("starts", sb);
        auto r      = m1.add_instruction(
            migraphx::make_op("slice",
                                   {{"axes", {1}},
                                    {"starts", migraphx::value::array{migraphx::to_value(dd{n})}},
                                    {"ends", {3}},
                                    {"mode", migraphx::value::array{"starts"}}}),
            data,
            starts);
        m1.add_return({r});
    }

    migraphx::module m2;
    {
        auto data   = m2.add_parameter("data", s);
        auto starts = m2.add_parameter("starts", sb);
        auto r      = m2.add_instruction(
            migraphx::make_op(
                "slice",
                {{"axes", {1}},
                      {"starts",
                       migraphx::value::array{migraphx::to_value(dd{migraphx::sym::min(n, lit(3))})}},
                      {"ends", {3}},
                      {"mode", migraphx::value::array{"starts"}}}),
            data,
            starts);
        m2.add_return({r});
    }
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(slice_sym_mixed_bounds_test)
{
    // One symbolic entry routes the whole attribute through the symbolic path. Each entry is
    // still clamped against its own axis length (3 for axis 1, 5 for axis 3), and the concrete
    // entry stays concrete.
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    migraphx::shape sb{migraphx::shape::int64_type, {2}};

    migraphx::module m1;
    {
        auto data = m1.add_parameter("data", s);
        auto ends = m1.add_parameter("ends", sb);
        auto r    = m1.add_instruction(
            migraphx::make_op("slice",
                                 {{"axes", {1, 3}},
                                  {"starts", {0, 0}},
                                  {"ends", migraphx::value::array{migraphx::to_value(dd{n}), 9}},
                                  {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m1.add_return({r});
    }

    migraphx::module m2;
    {
        auto data = m2.add_parameter("data", s);
        auto ends = m2.add_parameter("ends", sb);
        auto r    = m2.add_instruction(
            migraphx::make_op(
                "slice",
                {{"axes", {1, 3}},
                    {"starts", {0, 0}},
                    {"ends",
                     migraphx::value::array{migraphx::to_value(dd{migraphx::sym::min(n, lit(3))}), 5}},
                    {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m2.add_return({r});
    }
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(slice_sym_symbolic_axis_len_test)
{
    // When the sliced axis is itself symbolic, the clamp bound is that axis's symbol instead of
    // a compile-time length.
    auto k = var("k", {2, 6});
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {dd{k}, dd{lit(4)}}};
    migraphx::shape sb{migraphx::shape::int64_type, {1}};

    migraphx::module m1;
    {
        auto data = m1.add_parameter("data", s);
        auto ends = m1.add_parameter("ends", sb);
        auto r    = m1.add_instruction(
            migraphx::make_op("slice",
                                 {{"axes", {0}},
                                  {"starts", {0}},
                                  {"ends", migraphx::value::array{migraphx::to_value(dd{n})}},
                                  {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m1.add_return({r});
    }

    migraphx::module m2;
    {
        auto data = m2.add_parameter("data", s);
        auto ends = m2.add_parameter("ends", sb);
        auto r    = m2.add_instruction(
            migraphx::make_op(
                "slice",
                {{"axes", {0}},
                    {"starts", {0}},
                    {"ends", migraphx::value::array{migraphx::to_value(dd{migraphx::sym::min(n, k)})}},
                    {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m2.add_return({r});
    }
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(slice_sym_normalize_idempotent_test)
{
    // Normalizing an already normalized symbolic bound must not nest a second clamp.
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    migraphx::shape sb{migraphx::shape::int64_type, {1}};

    migraphx::module m1;
    {
        auto data = m1.add_parameter("data", s);
        auto ends = m1.add_parameter("ends", sb);
        auto r    = m1.add_instruction(
            migraphx::make_op("slice",
                                 {{"axes", {3}},
                                  {"starts", {0}},
                                  {"ends", migraphx::value::array{migraphx::to_value(dd{n})}},
                                  {"mode", migraphx::value::array{"ends"}}}),
            data,
            ends);
        m1.add_return({r});
    }
    run_pass(m1);
    auto once = m1;
    run_pass(m1);

    EXPECT(m1 == once);
}

TEST_CASE(slice_sym_missing_axes_throws)
{
    // The axes come from a variable input, leaving the axes attribute empty, so there is no
    // axis to pair the symbolic bound with.
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    migraphx::shape sb{migraphx::shape::int64_type, {1}};

    EXPECT(test::throws<migraphx::exception>(
        [&] {
            migraphx::module m;
            auto data    = m.add_parameter("data", s);
            auto ends_in = m.add_parameter("ends", sb);
            auto axes_in = m.add_parameter("axes", sb);
            m.add_instruction(
                migraphx::make_op("slice",
                                  {{"starts", {0}},
                                   {"ends", migraphx::value::array{migraphx::to_value(dd{n})}},
                                   {"mode", migraphx::value::array{"ends", "axes"}}}),
                data,
                ends_in,
                axes_in);
        },
        "symbolic bounds require one axis per bound"));
}

TEST_CASE(slice_sym_nonfixed_axis_throws)
{
    // A range-based dynamic axis is neither symbolic nor fixed, so it has no length expression
    // to clamp the symbolic bound against.
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {3, 3}}};
    migraphx::shape sb{migraphx::shape::int64_type, {1}};

    EXPECT(test::throws<migraphx::exception>(
        [&] {
            migraphx::module m;
            auto data = m.add_parameter("data", s);
            auto ends = m.add_parameter("ends", sb);
            m.add_instruction(
                migraphx::make_op("slice",
                                  {{"axes", {0}},
                                   {"starts", {0}},
                                   {"ends", migraphx::value::array{migraphx::to_value(dd{n})}},
                                   {"mode", migraphx::value::array{"ends"}}}),
                data,
                ends);
        },
        "cannot normalize a symbolic bound on a non-fixed axis"));
}

static migraphx::module create_test_op(const std::vector<int64_t>& axes)
{
    migraphx::module m;
    migraphx::shape sd{migraphx::shape::float_type, {2, 3, 4}};
    auto di = m.add_parameter("data", sd);
    auto r  = m.add_instruction(normalize_test_op{axes}, di);
    m.add_return({r});

    return m;
}

TEST_CASE(test_op)
{
    std::vector<int64_t> axes1 = {-4, 5};
    auto m1                    = create_test_op(axes1);

    std::vector<int64_t> axes2 = {1, 2};
    auto m2                    = create_test_op(axes2);

    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
