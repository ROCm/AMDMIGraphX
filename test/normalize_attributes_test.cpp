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

// Characterizes the integer normalization surface of normalize_attributes: the flag handling in
// tune_attribute reached through the normalize_axes and normalize_indices entry points, and the
// operator-level entry point covering padding, the normalize_axes map, and the scalar attribute
// path. Symbolic normalization is covered by the dyn_slice cases in normalize_ops_test.cpp.

#include <migraphx/functional.hpp>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>

#include <test.hpp>

using na   = migraphx::op::normalize_attribute;
using ints = std::vector<int64_t>;

static migraphx::shape rank4() { return {migraphx::shape::float_type, {2, 3, 4, 5}}; }

// Clip on both ends and treat both bounds as inclusive, so a value is pinned to its axis length
// rather than range checked. This is the combination the slice bounds use.
static migraphx::value clip_inclusive(bool use_len)
{
    if(use_len)
        return migraphx::value::array{
            na::use_len, na::clip_max, na::include_max, na::clip_min, na::include_min};
    return migraphx::value::array{na::clip_max, na::include_max, na::clip_min, na::include_min};
}

// ===================================================================
// normalize_axes: no axes are passed, so every value uses the rank
// ===================================================================

TEST_CASE(normalize_axes_negative_resolved_against_rank)
{
    EXPECT(migraphx::normalize_axes({-1, 0, 3}, rank4(), migraphx::value::array{na::include_min}) ==
           ints{3, 0, 3});
}

TEST_CASE(normalize_axes_empty_is_unchanged)
{
    EXPECT(migraphx::normalize_axes({}, rank4(), migraphx::value::array{na::include_min}).empty());
}

TEST_CASE(normalize_axes_above_rank_throws)
{
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::normalize_axes({4}, rank4(), migraphx::value::array{na::include_min}); },
        "value out of range!"));
}

TEST_CASE(normalize_axes_below_min_throws)
{
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::normalize_axes({-5}, rank4(), migraphx::value::array{na::include_min}); },
        "attribute out of range!"));
}

TEST_CASE(normalize_axes_include_max_allows_the_rank)
{
    EXPECT(migraphx::normalize_axes(
               {4}, rank4(), migraphx::value::array{na::include_min, na::include_max}) == ints{4});
}

TEST_CASE(normalize_axes_use_output_extends_the_rank)
{
    // use_output raises the maximum to the rank plus the number of values being normalized.
    auto o = migraphx::value::array{na::use_output, na::include_min};
    EXPECT(migraphx::normalize_axes({4}, rank4(), o) == ints{4});
    EXPECT(migraphx::normalize_axes({-1}, rank4(), o) == ints{4});
    // Two values raise it to 6, so -1 resolves to 5.
    EXPECT(migraphx::normalize_axes({5, -1}, rank4(), o) == ints{5, 5});
}

TEST_CASE(normalize_axes_clip_max_excludes_the_maximum)
{
    // Without include_max the clip limit is one below the rank.
    EXPECT(migraphx::normalize_axes(
               {10},
               rank4(),
               migraphx::value::array{na::clip_max, na::clip_min, na::include_min}) == ints{3});
    // With it the rank itself is the limit.
    EXPECT(migraphx::normalize_axes({10}, rank4(), clip_inclusive(false)) == ints{4});
}

TEST_CASE(normalize_axes_clip_min_excludes_the_minimum)
{
    // Without include_min the clip limit is one above -rank, so it lands on -3, then 1.
    EXPECT(migraphx::normalize_axes(
               {-10}, rank4(), migraphx::value::array{na::clip_max, na::clip_min}) == ints{1});
    // With it the limit is -rank, which resolves to 0.
    EXPECT(migraphx::normalize_axes({-10}, rank4(), clip_inclusive(false)) == ints{0});
}

TEST_CASE(normalize_axes_message_carries_the_prefix)
{
    EXPECT(test::throws<migraphx::exception>(
        [&] {
            migraphx::normalize_axes(
                {4}, rank4(), migraphx::value::array{na::include_min}, "MY_OP: ");
        },
        "MY_OP: value out of range!"));
}

TEST_CASE(normalize_axes_use_len_has_no_axes_to_look_up)
{
    // normalize_axes passes no axes, so use_len finds nothing to index and the rank is kept.
    EXPECT(migraphx::normalize_axes({10}, rank4(), clip_inclusive(true)) == ints{4});
}

// ===================================================================
// normalize_indices: axes select the length each value is bound to
// ===================================================================

TEST_CASE(normalize_indices_use_len_clips_per_axis)
{
    // Axis 1 has length 3 and axis 3 has length 5, so each value takes its own limit.
    EXPECT(migraphx::normalize_indices({10, 10}, {1, 3}, rank4(), clip_inclusive(true)) ==
           ints{3, 5});
    EXPECT(migraphx::normalize_indices({2, 3}, {1, 3}, rank4(), clip_inclusive(true)) ==
           ints{2, 3});
}

TEST_CASE(normalize_indices_use_len_resolves_negatives_per_axis)
{
    EXPECT(migraphx::normalize_indices({-1, -2}, {1, 3}, rank4(), clip_inclusive(true)) ==
           ints{2, 3});
}

TEST_CASE(normalize_indices_without_use_len_uses_the_rank)
{
    EXPECT(migraphx::normalize_indices({10}, {3}, rank4(), clip_inclusive(false)) == ints{4});
}

TEST_CASE(normalize_indices_more_axes_than_values_throws)
{
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::normalize_indices({0}, {0, 1}, rank4(), clip_inclusive(true)); },
        "more axes than values to normalize!"));
}

TEST_CASE(normalize_indices_extra_values_fall_back_to_the_rank)
{
    // Only the first axes.size() entries take a length; the rest keep the rank as their maximum.
    EXPECT(migraphx::normalize_indices({10, 10}, {3}, rank4(), clip_inclusive(true)) == ints{5, 4});
}

TEST_CASE(normalize_indices_dynamic_fixed_axis_uses_the_interval_max)
{
    migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {3, 3}}};
    EXPECT(migraphx::normalize_indices({10}, {1}, s, clip_inclusive(true)) == ints{3});
}

TEST_CASE(normalize_indices_dynamic_nonfixed_axis_is_unchanged)
{
    // With no single length to normalize against the values are returned as given, so the caller
    // has to renormalize once the dimensions are known.
    migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {3, 3}}};
    EXPECT(migraphx::normalize_indices({10}, {0}, s, clip_inclusive(true)) == ints{10});
    EXPECT(migraphx::normalize_indices({-10}, {0}, s, clip_inclusive(true)) == ints{-10});
}

// ===================================================================
// normalize_attributes: the operator level entry point
// ===================================================================

struct padding_test_op
{
    std::vector<std::size_t> padding          = {};
    migraphx::op::padding_mode_t padding_mode = migraphx::op::padding_mode_t::default_;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.padding, "padding"), f(self.padding_mode, "padding_mode"));
    }

    migraphx::value attributes() const { return {{"normalize_padding", "padding"}}; }
    std::string name() const { return "normalize_attributes_test::padding_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
};

struct axes_test_op
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
        normalize["axes"] = migraphx::value::array{na::include_min};
        return {{"normalize_axes", normalize}};
    }
    std::string name() const { return "normalize_attributes_test::axes_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
};

// Declares axes first so it is resolved before the bound that is normalized against it.
struct axes_and_starts_test_op
{
    std::vector<int64_t> axes   = {};
    std::vector<int64_t> starts = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.axes, "axes"), f(self.starts, "starts"));
    }

    migraphx::value attributes() const
    {
        migraphx::value normalize;
        normalize["axes"]   = migraphx::value::array{na::include_min};
        normalize["starts"] = migraphx::value::array{
            na::use_len, na::clip_max, na::include_max, na::clip_min, na::include_min};
        return {{"normalize_axes", normalize}};
    }
    std::string name() const { return "normalize_attributes_test::axes_and_starts_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
};

struct scalar_axis_test_op
{
    int64_t axis = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.axis, "axis"));
    }

    migraphx::value attributes() const
    {
        migraphx::value normalize;
        normalize["axis"] = migraphx::value::array{na::include_min};
        return {{"normalize_axes", normalize}};
    }
    std::string name() const { return "normalize_attributes_test::scalar_axis_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
};

// Names a key in normalize_axes that the operator does not actually have.
struct missing_key_test_op
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
        normalize["starts"] = migraphx::value::array{na::include_min};
        return {{"normalize_axes", normalize}};
    }
    std::string name() const { return "normalize_attributes_test::missing_key_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
};

struct plain_test_op
{
    std::vector<int64_t> axes = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.axes, "axes"));
    }

    std::string name() const { return "normalize_attributes_test::plain_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
};

TEST_CASE(normalize_padding_doubles_a_one_sided_attribute)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::operation op = padding_test_op{{1, 2}};
    EXPECT(migraphx::normalize_attributes(op, s));
    EXPECT(op.to_value().at("padding").to_vector<std::size_t>() ==
           std::vector<std::size_t>{1, 2, 1, 2});
}

TEST_CASE(normalize_padding_leaves_a_two_sided_attribute)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::operation op = padding_test_op{{1, 2, 3, 4}};
    EXPECT(migraphx::normalize_attributes(op, s));
    EXPECT(op.to_value().at("padding").to_vector<std::size_t>() ==
           std::vector<std::size_t>{1, 2, 3, 4});
}

TEST_CASE(normalize_padding_inconsistent_size_throws)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::operation op = padding_test_op{{1, 2, 3}};
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::normalize_attributes(op, s); },
                                             "inconsistent padding vector size"));
}

TEST_CASE(normalize_padding_auto_mode_is_left_to_the_target)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::operation op = padding_test_op{{1, 2}, migraphx::op::padding_mode_t::same_upper};
    EXPECT(not migraphx::normalize_attributes(op, s));
    EXPECT(op.to_value().at("padding").to_vector<std::size_t>() == std::vector<std::size_t>{1, 2});
}

TEST_CASE(normalize_attributes_reports_whether_it_tuned)
{
    migraphx::operation tuned = axes_test_op{{-1}};
    EXPECT(migraphx::normalize_attributes(tuned, rank4()));
    EXPECT(tuned.to_value().at("axes").to_vector<int64_t>() == ints{3});

    migraphx::operation untuned = plain_test_op{{-1}};
    EXPECT(not migraphx::normalize_attributes(untuned, rank4()));
    EXPECT(untuned.to_value().at("axes").to_vector<int64_t>() == ints{-1});
}

TEST_CASE(normalize_attributes_resolves_axes_before_a_dependent_key)
{
    // The keys normalize in declaration order, so starts is clipped against the resolved axis 3
    // (length 5) rather than the original -1.
    migraphx::operation op = axes_and_starts_test_op{{-1}, {10}};
    EXPECT(migraphx::normalize_attributes(op, rank4()));
    EXPECT(op.to_value().at("axes").to_vector<int64_t>() == ints{3});
    EXPECT(op.to_value().at("starts").to_vector<int64_t>() == ints{5});
}

TEST_CASE(normalize_attributes_scalar_attribute)
{
    // A non-array attribute normalizes as a single value, passing itself as its own axes.
    migraphx::operation op = scalar_axis_test_op{-1};
    EXPECT(migraphx::normalize_attributes(op, rank4()));
    EXPECT(op.to_value().at("axis").to<int64_t>() == 3);
}

TEST_CASE(normalize_attributes_missing_key_throws)
{
    migraphx::operation op = missing_key_test_op{{0}};
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::normalize_attributes(op, rank4()); },
                                             "\"starts\" not exist!"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
