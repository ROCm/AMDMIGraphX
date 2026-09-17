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
#include <migraphx/reshape_dims.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/errors.hpp>

#include <test.hpp>

using dd = migraphx::shape::dynamic_dimension;
using se = migraphx::sym::expr;
using migraphx::sym::lit;
using migraphx::sym::var;

static const auto ftype = migraphx::shape::float_type;

// reshape_dims always answers in the symbolic domain, so evaluate the result back to a concrete
// shape for comparison. nullopt means the layout could not be proven.
static migraphx::optional<migraphx::shape>
static_reshape(const migraphx::shape& input, const std::vector<std::size_t>& rdims, bool lazy)
{
    auto r = migraphx::reshape_dims(input, rdims, {.lazy = lazy});
    if(not r.has_value())
        return migraphx::nullopt;
    return r->to_static();
}

static migraphx::optional<migraphx::shape>
sym_reshape(const migraphx::shape& input,
            const std::vector<se>& rdims,
            bool lazy,
            const std::unordered_map<se, std::size_t>& sym_map)
{
    auto r = migraphx::reshape_dims(input, rdims, {.lazy = lazy});
    if(not r.has_value())
        return migraphx::nullopt;
    return r->to_static(sym_map);
}

////////////////////////////////////////////////////////////////////////////////
// reshape_dims: static inputs
////////////////////////////////////////////////////////////////////////////////

TEST_CASE(standard_merge)
{
    migraphx::shape s{ftype, {2, 3, 4}};
    migraphx::shape expected{ftype, {2, 12}};
    EXPECT(static_reshape(s, {2, 12}, true) == expected);
    EXPECT(static_reshape(s, {2, 12}, false) == expected);
}

TEST_CASE(standard_split)
{
    migraphx::shape s{ftype, {2, 12}};
    migraphx::shape expected{ftype, {2, 3, 4}};
    EXPECT(static_reshape(s, {2, 3, 4}, true) == expected);
    EXPECT(static_reshape(s, {2, 3, 4}, false) == expected);
}

TEST_CASE(standard_identity)
{
    migraphx::shape s{ftype, {2, 3, 4}};
    EXPECT(static_reshape(s, {2, 3, 4}, true) == s);
}

TEST_CASE(standard_flatten)
{
    migraphx::shape s{ftype, {2, 3, 4}};
    migraphx::shape expected{ftype, {24}};
    EXPECT(static_reshape(s, {24}, true) == expected);
}

// Merging axes that are not adjacent in memory cannot be expressed as a view, so lazy reshape
// declines while a copy-permitting reshape repacks to a standard layout.
TEST_CASE(transposed_unmergeable)
{
    migraphx::shape s{ftype, {2, 3, 4}, {12, 1, 3}};
    migraphx::shape expected{ftype, {2, 12}};
    EXPECT(static_reshape(s, {2, 12}, true) == migraphx::nullopt);
    EXPECT(static_reshape(s, {2, 12}, false) == expected);
}

// The trailing axes of this permutation are adjacent, so the merge holds as a view and the
// permutation carries through to the result.
TEST_CASE(transposed_mergeable)
{
    migraphx::shape s{ftype, {2, 3, 4}, {1, 8, 2}};
    migraphx::shape expected{ftype, {2, 12}, {1, 2}};
    EXPECT(static_reshape(s, {2, 12}, true) == expected);
    EXPECT(static_reshape(s, {2, 12}, false) == expected);
}

// Splitting an axis is the inverse of merging it and recovers the original strides.
TEST_CASE(transposed_split)
{
    migraphx::shape s{ftype, {2, 12}, {1, 2}};
    migraphx::shape expected{ftype, {2, 3, 4}, {1, 8, 2}};
    EXPECT(static_reshape(s, {2, 3, 4}, true) == expected);
}

// A broadcasted axis keeps its zero stride through a lazy merge of the packed trailing axes.
// Without a view requirement the ambiguous permutation falls back to a standard layout instead.
TEST_CASE(broadcasted)
{
    migraphx::shape s{ftype, {2, 3, 4}, {0, 4, 1}};
    migraphx::shape lazy_expected{ftype, {2, 12}, {0, 1}};
    migraphx::shape copy_expected{ftype, {2, 12}};
    EXPECT(static_reshape(s, {2, 12}, true) == lazy_expected);
    EXPECT(static_reshape(s, {2, 12}, false) == copy_expected);
}

TEST_CASE(broadcasted_scalar)
{
    migraphx::shape s{ftype, {2, 3}, {0, 0}};
    migraphx::shape expected{ftype, {6}, {0}};
    EXPECT(static_reshape(s, {6}, true) == expected);
}

// A broadcast axis cannot merge into a non-broadcast one, since the result would need two
// different strides for one axis.
TEST_CASE(broadcasted_unmergeable)
{
    migraphx::shape s{ftype, {2, 3}, {0, 1}};
    EXPECT(static_reshape(s, {6}, true) == migraphx::nullopt);
}

// A sliced shape has gaps between its axes, so merging them loses the gap and needs a copy.
TEST_CASE(nonpacked)
{
    migraphx::shape s{ftype, {2, 2}, {4, 1}};
    migraphx::shape expected{ftype, {4}};
    EXPECT(static_reshape(s, {4}, true) == migraphx::nullopt);
    EXPECT(static_reshape(s, {4}, false) == expected);
}

// Axes of length 1 past the end of the walk inherit the last stride.
TEST_CASE(trailing_ones)
{
    migraphx::shape s{ftype, {2, 3, 4}, {1, 8, 2}};
    migraphx::shape expected{ftype, {2, 12, 1, 1}, {1, 2, 2, 2}};
    EXPECT(static_reshape(s, {2, 12, 1, 1}, true) == expected);
}

// A trailing axis that is not 1 would change the element count.
TEST_CASE(trailing_non_one)
{
    migraphx::shape s{ftype, {2, 3, 4}, {1, 8, 2}};
    EXPECT(static_reshape(s, {2, 12, 2}, true) == migraphx::nullopt);
}

// No run of input axes multiplies to 5, so the walk cannot line the two shapes up.
TEST_CASE(mismatched_elements)
{
    migraphx::shape s{ftype, {2, 3, 4}, {12, 1, 3}};
    EXPECT(static_reshape(s, {5, 5}, true) == migraphx::nullopt);
    EXPECT(static_reshape(s, {5, 5}, false) == migraphx::nullopt);
}

// Range-based dynamic dimensions have no stride expressions to reason about, so the layout is
// unprovable rather than an error.
TEST_CASE(range_dynamic)
{
    migraphx::shape s{ftype, {{1, 4}, {3, 3}, {4, 4}}};
    EXPECT(static_reshape(s, {2, 12}, true) == migraphx::nullopt);
    EXPECT(static_reshape(s, {2, 12}, false) == migraphx::nullopt);
}

// A static input and its symbolic lift must resolve through the same path. Static shapes carry no
// dyn_dims()/dyn_strides(), so they have to be lifted internally rather than read as if they were
// already symbolic.
TEST_CASE(static_matches_symbolic_lift)
{
    const std::vector<migraphx::shape> inputs           = {{ftype, {2, 3, 4}},
                                                           {ftype, {2, 3, 4}, {12, 1, 3}},
                                                           {ftype, {2, 3, 4}, {1, 8, 2}},
                                                           {ftype, {2, 3, 4}, {0, 4, 1}},
                                                           {ftype, {2, 3, 4}, {24, 8, 2}}};
    const std::vector<std::vector<std::size_t>> targets = {
        {2, 12}, {24}, {2, 3, 4}, {6, 4}, {2, 2, 6}};
    for(const auto& s : inputs)
    {
        for(const auto& target : targets)
        {
            for(bool lazy : {true, false})
            {
                EXPECT(static_reshape(s, target, lazy) ==
                       static_reshape(s.to_symbolic(), target, lazy));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// reshape_dims: symbolic inputs
////////////////////////////////////////////////////////////////////////////////

TEST_CASE(symbolic_standard)
{
    auto n                                      = var("n", {1, 8});
    std::unordered_map<se, std::size_t> sym_map = {{n, 2}};
    migraphx::shape s{ftype, {dd{n}, dd{lit(3)}, dd{lit(4)}}};
    migraphx::shape expected{ftype, {2, 12}};
    EXPECT(sym_reshape(s, {n, lit(12)}, true, sym_map) == expected);
}

// A symbolic stride merges the same way a literal one does when the ratio is provable.
TEST_CASE(symbolic_transposed_mergeable)
{
    auto n                                      = var("n", {1, 8});
    std::unordered_map<se, std::size_t> sym_map = {{n, 2}};
    migraphx::shape s{ftype, {dd{n}, dd{lit(3)}, dd{lit(4)}}, {lit(1), n * 4, n}};
    migraphx::shape expected{ftype, {2, 12}, {1, 2}};
    EXPECT(sym_reshape(s, {n, lit(12)}, true, sym_map) == expected);
}

TEST_CASE(symbolic_broadcasted)
{
    auto n                                      = var("n", {1, 8});
    std::unordered_map<se, std::size_t> sym_map = {{n, 2}};
    migraphx::shape s{ftype, {dd{n}, dd{lit(3)}, dd{lit(4)}}, {lit(0), lit(4), lit(1)}};
    migraphx::shape lazy_expected{ftype, {2, 12}, {0, 1}};
    migraphx::shape copy_expected{ftype, {2, 12}};
    EXPECT(sym_reshape(s, {n, lit(12)}, true, sym_map) == lazy_expected);
    EXPECT(sym_reshape(s, {n, lit(12)}, false, sym_map) == copy_expected);
}

// n ranges over [1, 8], so neither n < 8 nor 8 < n holds for every value and the walk cannot pick
// between squeezing and unsqueezing.
TEST_CASE(symbolic_unprovable_ordering)
{
    auto n = var("n", {1, 8});
    migraphx::shape s{ftype, {dd{n}, dd{lit(4)}}, {lit(1), n}};
    EXPECT(migraphx::reshape_dims(s, {lit(8), lit(4)}, {.lazy = true}) == migraphx::nullopt);
}

// Merging a literal axis with a symbolic one yields a symbolic output dim. n is bounded below by
// 2 so that 4 < 4n is provable; at n == 1 the ordering would be indeterminate.
TEST_CASE(symbolic_merge_into_symbol)
{
    auto n                                      = var("n", {2, 8});
    std::unordered_map<se, std::size_t> sym_map = {{n, 2}};
    migraphx::shape s{ftype, {dd{lit(3)}, dd{lit(4)}, dd{n}}, {lit(1), n * 3, lit(3)}};
    migraphx::shape expected{ftype, {3, 8}, {1, 3}};
    EXPECT(sym_reshape(s, {lit(3), n * 4}, true, sym_map) == expected);
}

// Column-major strides make the outer axis the denser one, so merging the two axes is a
// transposed merge that no view can express. A copy repacks to a standard layout.
TEST_CASE(symbolic_transposed_unmergeable)
{
    auto n                                      = var("n", {1, 8});
    std::unordered_map<se, std::size_t> sym_map = {{n, 3}};
    migraphx::shape s{ftype, {dd{n}, dd{lit(4)}}, {lit(1), n}};
    migraphx::shape expected{ftype, {12}};
    EXPECT(sym_reshape(s, {n * 4}, true, sym_map) == migraphx::nullopt);
    EXPECT(sym_reshape(s, {n * 4}, false, sym_map) == expected);
}

// Resolving the symbols first and reshaping the concrete shape must agree with reshaping
// symbolically and resolving afterwards.
TEST_CASE(symbolic_matches_static_eval)
{
    auto n                                      = var("n", {1, 8});
    std::unordered_map<se, std::size_t> sym_map = {{n, 2}};
    const std::vector<migraphx::shape> inputs   = {
        {ftype, {dd{n}, dd{lit(3)}, dd{lit(4)}}},
        {ftype, {dd{n}, dd{lit(3)}, dd{lit(4)}}, {lit(1), n * 4, n}},
        {ftype, {dd{n}, dd{lit(3)}, dd{lit(4)}}, {lit(0), lit(4), lit(1)}},
        {ftype, {dd{n}, dd{lit(3)}, dd{lit(4)}}, {lit(12), lit(1), lit(3)}}};
    for(const auto& s : inputs)
    {
        for(bool lazy : {true, false})
        {
            auto from_sym    = sym_reshape(s, {n, lit(12)}, lazy, sym_map);
            auto from_static = static_reshape(s.to_static(sym_map), {2, 12}, lazy);
            EXPECT(from_sym == from_static);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// resolve_reshape_dims
////////////////////////////////////////////////////////////////////////////////

TEST_CASE(resolve_literals)
{
    migraphx::shape s{ftype, {dd{lit(2)}, dd{lit(3)}, dd{lit(4)}}};
    std::vector<dd> expected = {dd{lit(2)}, dd{lit(12)}};
    EXPECT(migraphx::resolve_reshape_dims(s, {2, 12}) == expected);
}

// A 0 entry copies the input dim at that index.
TEST_CASE(resolve_zero_copies_input_dim)
{
    migraphx::shape s{ftype, {dd{lit(2)}, dd{lit(3)}, dd{lit(4)}}};
    std::vector<dd> expected = {dd{lit(2)}, dd{lit(12)}};
    EXPECT(migraphx::resolve_reshape_dims(s, {0, 12}) == expected);
}

TEST_CASE(resolve_zero_copies_symbol)
{
    auto n = var("n", {1, 8});
    migraphx::shape s{ftype, {dd{n}, dd{lit(12)}}};
    std::vector<dd> expected = {dd{n}, dd{lit(12)}};
    EXPECT(migraphx::resolve_reshape_dims(s, {0, 12}) == expected);
}

// A -1 entry is the leftover element count after the explicit dims.
TEST_CASE(resolve_infers_negative_one)
{
    migraphx::shape s{ftype, {dd{lit(2)}, dd{lit(3)}, dd{lit(4)}}};
    std::vector<dd> expected = {dd{lit(2)}, dd{lit(12)}};
    EXPECT(migraphx::resolve_reshape_dims(s, {2, -1}) == expected);
    EXPECT(migraphx::resolve_reshape_dims(s, {-1, 12}) == expected);
}

TEST_CASE(resolve_infers_negative_one_over_symbol)
{
    auto n = var("n", {1, 8});
    migraphx::shape s{ftype, {dd{n}, dd{lit(3)}, dd{lit(4)}}};
    auto result = migraphx::resolve_reshape_dims(s, {-1, 12});
    EXPECT(result.size() == 2);
    EXPECT(result[0] == dd{n});
    EXPECT(result[1] == dd{lit(12)});
}

// A symbolic dim entry is taken as-is.
TEST_CASE(resolve_symbolic_entry)
{
    auto n = var("n", {1, 8});
    migraphx::shape s{ftype, {dd{n}, dd{lit(3)}, dd{lit(4)}}};
    std::vector<dd> expected = {dd{n}, dd{lit(12)}};
    EXPECT(migraphx::resolve_reshape_dims(s, {migraphx::dim_like{dd{n}}, 12}) == expected);
}

TEST_CASE(resolve_rank_change)
{
    migraphx::shape s{ftype, {dd{lit(24)}}};
    std::vector<dd> expected = {dd{lit(2)}, dd{lit(3)}, dd{lit(4)}};
    EXPECT(migraphx::resolve_reshape_dims(s, {2, 3, -1}) == expected);
}

////////////////////////////////////////////////////////////////////////////////
// validate_reshape_dims
////////////////////////////////////////////////////////////////////////////////

TEST_CASE(validate_accepts_literals)
{
    migraphx::validate_reshape_dims("reshape", {1, 2, 3});
    migraphx::validate_reshape_dims("reshape", {0, 2, -1});
    migraphx::validate_reshape_dims("reshape", {});
}

TEST_CASE(validate_accepts_symbolic)
{
    auto n = var("n", {1, 8});
    migraphx::validate_reshape_dims("reshape", {migraphx::dim_like{dd{n}}, 12});
}

TEST_CASE(validate_rejects_range_dim)
{
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::validate_reshape_dims("reshape", {migraphx::dim_like{dd{1, 4}}, 12}); },
        "dim entries must be int64 or symbolic"));
}

TEST_CASE(validate_rejects_multiple_inferred_dims)
{
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::validate_reshape_dims("reshape", {-1, 2, -1}); },
        "can only have one -1 dim"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
