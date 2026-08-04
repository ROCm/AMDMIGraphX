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
#include <migraphx/dim_like.hpp>
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

// A bound attribute that can hold a symbolic value has to declare use_sym, otherwise its
// normalized value could not be stored back. This operator deliberately leaves it out.
struct no_use_sym_test_op
{
    std::vector<int64_t> axes             = {};
    std::vector<migraphx::dim_like> bound = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.axes, "axes"), f(self.bound, "bound"));
    }

    migraphx::value attributes() const
    {
        migraphx::value normalize;
        normalize["bound"] = migraphx::value::array{migraphx::op::normalize_attribute::clip_max,
                                                    migraphx::op::normalize_attribute::clip_min,
                                                    migraphx::op::normalize_attribute::include_max,
                                                    migraphx::op::normalize_attribute::use_len,
                                                    migraphx::op::normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "normalize_ops_test::no_use_sym_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
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

// dyn_slice always takes its bounds as inputs, and its starts/ends attributes describe those
// inputs at compile time. Only the attributes are normalized here.
static migraphx::module create_dyn_slice(const migraphx::shape& data_shape,
                                         const migraphx::value& attributes,
                                         std::size_t nbounds = 1)
{
    migraphx::module m;
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {nbounds}};
    auto data   = m.add_parameter("data", data_shape);
    auto starts = m.add_parameter("starts", bounds_shape);
    auto ends   = m.add_parameter("ends", bounds_shape);
    auto r      = m.add_instruction(migraphx::make_op("dyn_slice", attributes), data, starts, ends);
    m.add_return({r});

    return m;
}

static migraphx::value sym_bound(const migraphx::sym::expr& e)
{
    return migraphx::value::array{migraphx::to_value(dd{e})};
}

TEST_CASE(dyn_slice_sym_ends_clamped_test)
{
    // n is not provably ordered against the axis length 5, so the bound clamps to min(n, 5).
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};

    auto m1 = create_dyn_slice(s, {{"axes", {3}}, {"starts", {0}}, {"ends", sym_bound(n)}});
    auto m2 = create_dyn_slice(
        s, {{"axes", {3}}, {"starts", {0}}, {"ends", sym_bound(migraphx::sym::min(n, lit(5)))}});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(dyn_slice_sym_ends_below_len_test)
{
    // n < 5 is provable, so the bound keeps the bare symbol rather than gaining a min wrapper.
    auto n = var("n", {1, 4});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};

    auto m1 = create_dyn_slice(s, {{"axes", {3}}, {"starts", {0}}, {"ends", sym_bound(n)}});
    auto m2 = create_dyn_slice(s, {{"axes", {3}}, {"starts", {0}}, {"ends", sym_bound(n)}});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(dyn_slice_sym_ends_at_len_test)
{
    // n >= 5 is provable, so the bound collapses to the axis length and demotes to an integer.
    auto n = var("n", {6, 9});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};

    auto m1 = create_dyn_slice(s, {{"axes", {3}}, {"starts", {0}}, {"ends", sym_bound(n)}});
    auto m2 = create_dyn_slice(s, {{"axes", {3}}, {"starts", {0}}, {"ends", {5}}});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(dyn_slice_sym_starts_clamped_test)
{
    // The starts attribute normalizes the same way. Axis 1 has length 3.
    auto n = var("n", {0, 5});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};

    auto m1 = create_dyn_slice(s, {{"axes", {1}}, {"starts", sym_bound(n)}, {"ends", {3}}});
    auto m2 = create_dyn_slice(
        s, {{"axes", {1}}, {"starts", sym_bound(migraphx::sym::min(n, lit(3)))}, {"ends", {3}}});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(dyn_slice_sym_mixed_bounds_test)
{
    // Each entry is clamped against its own axis length (3 for axis 1, 5 for axis 3), and the
    // concrete entry stays concrete.
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};

    auto m1 = create_dyn_slice(s,
                               {{"axes", {1, 3}},
                                {"starts", {0, 0}},
                                {"ends", migraphx::value::array{migraphx::to_value(dd{n}), 9}}},
                               2);
    auto m2 = create_dyn_slice(
        s,
        {{"axes", {1, 3}},
         {"starts", {0, 0}},
         {"ends",
          migraphx::value::array{migraphx::to_value(dd{migraphx::sym::min(n, lit(3))}), 5}}},
        2);
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(dyn_slice_sym_symbolic_axis_len_test)
{
    // When the sliced axis is itself symbolic, the clamp bound is that axis's symbol instead of
    // a compile-time length.
    auto k = var("k", {2, 6});
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {dd{k}, dd{lit(4)}}};

    auto m1 = create_dyn_slice(s, {{"axes", {0}}, {"starts", {0}}, {"ends", sym_bound(n)}});
    auto m2 = create_dyn_slice(
        s, {{"axes", {0}}, {"starts", {0}}, {"ends", sym_bound(migraphx::sym::min(n, k))}});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(dyn_slice_concrete_bounds_symbolic_axis_len_test)
{
    // A concrete bound still normalizes symbolically when the axis it is clamped against is a
    // symbol: end 2 is not provably below k, so it becomes min(2, k).
    auto k = var("k", {1, 6});
    migraphx::shape s{migraphx::shape::float_type, {dd{k}, dd{lit(4)}}};

    auto m1 = create_dyn_slice(s, {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}});
    auto m2 = create_dyn_slice(
        s, {{"axes", {0}}, {"starts", {0}}, {"ends", sym_bound(migraphx::sym::min(lit(2), k))}});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(dyn_slice_sym_normalize_idempotent_test)
{
    // Normalizing an already normalized symbolic bound must not nest a second clamp.
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};

    auto m1 = create_dyn_slice(s, {{"axes", {3}}, {"starts", {0}}, {"ends", sym_bound(n)}});
    run_pass(m1);
    auto once = m1;
    run_pass(m1);

    EXPECT(m1 == once);
}

TEST_CASE(dyn_slice_sym_indeterminate_sign_throws)
{
    // A bound that may be negative cannot be resolved into a from-the-end index.
    auto n = var("n", {-2, 2});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};

    EXPECT(test::throws<migraphx::exception>(
        [&] { create_dyn_slice(s, {{"axes", {3}}, {"starts", {0}}, {"ends", sym_bound(n)}}); },
        "bound of indeterminate sign cannot be normalized"));
}

TEST_CASE(dyn_slice_sym_nonfixed_axis_throws)
{
    // A range-based dynamic axis is neither symbolic nor fixed, so it has no length expression
    // to clamp the bound against.
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {3, 3}}};

    EXPECT(test::throws<migraphx::exception>(
        [&] { create_dyn_slice(s, {{"axes", {0}}, {"starts", {0}}, {"ends", sym_bound(n)}}); },
        "cannot normalize against a non-fixed axis"));
}

TEST_CASE(sym_value_without_use_sym_throws)
{
    // Normalizing a symbolic value into an attribute that did not opt in is rejected instead of
    // silently leaving the bound unnormalized.
    auto n = var("n", {1, 8});
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};

    EXPECT(test::throws<migraphx::exception>(
        [&] {
            migraphx::module m;
            auto data = m.add_parameter("data", s);
            m.add_instruction(no_use_sym_test_op{{3}, {dd{n}}}, data);
        },
        "symbolic values are not supported"));
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
