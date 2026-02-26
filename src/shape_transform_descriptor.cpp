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
 *
 */
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/output_iterator.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/erase.hpp>
#include <migraphx/common_dims.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/transform_view.hpp>
#include <map>
#include <unordered_set>
#include <deque>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using dimension = shape_transform_descriptor::dimension;

template <class Iterator, class Projection>
static auto compute_end_dim(Iterator start, Iterator last, std::size_t dim, Projection proj)
{
    std::size_t x = 1;
    auto it       = std::find_if(start, last, [&](auto d) {
        x *= proj(d);
        return x == dim;
    });
    if(it != last)
        return it;
    return start;
}

[[maybe_unused]] static void debug_print(const std::vector<dimension::sub>& subs)
{
    std::cout << '[' << stream_range(subs) << "]\n";
}
[[maybe_unused]] static void debug_print(const dimension& dim) { debug_print(dim.subdimensions); }
[[maybe_unused]] static void debug_print(const std::vector<dimension>& dims)
{
    stream_write_value(std::cout, dims);
    std::cout << std::endl;
}

shape_transform_descriptor::shape_transform_descriptor(const std::vector<std::size_t>& dims)
    : rank(dims.size())
{
    transform(dims,
              range(dims.size()),
              std::back_inserter(dimensions),
              [](std::size_t d, std::size_t a) -> dimension {
                  return {{dimension::sub{d, {a}}}};
              });
}

template <class Dimensions, class F>
static auto for_each_subdimension(Dimensions&& dimensions,
                                  F f) -> decltype(dimensions.begin()->subdimensions, void())
{
    for(auto& dim : dimensions)
    {
        for(auto& s : dim.subdimensions)
        {
            f(s);
        }
    }
}

template <class SubDimensions, class F>
static auto for_each_subdimension(SubDimensions&& subdimensions,
                                  F f) -> decltype(subdimensions.begin()->axis, void())
{
    for(auto& s : subdimensions)
    {
        f(s);
    }
}

static std::vector<dimension::sub> get_all_subdimensions(const std::vector<dimension>& dimensions)
{
    std::vector<dimension::sub> result;
    for(const auto& dim : dimensions)
    {
        result.insert(result.end(), dim.subdimensions.begin(), dim.subdimensions.end());
    }
    return result;
}

template <class Vector>
static std::vector<dimension::sub*> get_pointer_subdimensions(Vector& v)
{
    std::vector<dimension::sub*> result;
    for_each_subdimension(v, [&](dimension::sub& s) { result.push_back(&s); });
    return result;
}

template <class Dimensions, class Range, class F>
static void for_each_subdimension(Dimensions&& dimensions, Range&& r, F f)
{
    auto start = r.begin();
    auto last  = r.end();
    for(auto& dim : dimensions)
    {
        for(auto& s : dim.subdimensions)
        {
            if(start == last)
                return;
            f(s, *start);
            start++;
        }
    }
}

static void set_origin_axis(dimension::sub& s, const std::vector<std::size_t>& axis)
{
    assert(s.axis.empty() or s.hidden_axis.empty());
    if(s.has_hidden_axis())
        s.hidden_axis = axis;
    else
        s.axis = axis;
}

// Group all axes into a map with a key of the axis and the value is vector of
// all subdimensions that have that axis.
template <class Dimensions>
static std::map<std::size_t, std::vector<dimension::sub*>> group_axes(Dimensions& dimensions)
{
    using sub =
        std::conditional_t<std::is_const<Dimensions>{}, const dimension::sub, dimension::sub>;
    std::map<std::size_t, std::vector<sub*>> axes_map;
    for_each_subdimension(dimensions, [&](auto& s) {
        if(s.origin_axis().empty())
            return;
        axes_map[s.origin_axis().front()].push_back(&s);
    });
    return axes_map;
}

// Renumber all axes while preserving the order of the axes
static void renumber_axes(std::map<std::size_t, std::vector<dimension::sub*>>& axes_map)
{
    for(auto&& p : axes_map)
    {
        const auto& axis = p.first;
        auto& subs       = p.second;
        if(subs.size() == 1)
        {
            set_origin_axis(*subs[0], {axis});
        }
        else
        {
            std::sort(subs.begin(), subs.end(), by(std::less<>{}, [](const dimension::sub* s) {
                          return s->origin_axis();
                      }));
            for(std::size_t i : range(subs.size()))
                set_origin_axis(*subs[i], {axis, i});
        }
    }
}
static void renumber_axes(std::vector<dimension>& dimensions)
{
    auto axes_map = group_axes(dimensions);
    renumber_axes(axes_map);
}

static void remove_empty_sub_dims(std::vector<dimension::sub>& subdimensions)
{
    subdimensions.erase(std::remove_if(subdimensions.begin(),
                                       subdimensions.end(),
                                       [&](const dimension::sub& d) {
                                           return d.len == 1 and d.origin_axis().empty();
                                       }),
                        subdimensions.end());
    if(subdimensions.empty())
        subdimensions.push_back({1, {}});
}

static std::size_t len(const std::vector<dimension::sub*>& subs)
{
    return transform_accumulate(
        subs.begin(), subs.end(), std::size_t{1}, std::multiplies<>{}, [](const dimension::sub* s) {
            return s->len;
        });
}

static std::size_t visible_len(const std::vector<dimension::sub*>& subs)
{
    return transform_accumulate(
        subs.begin(), subs.end(), std::size_t{1}, std::multiplies<>{}, [](const dimension::sub* s) {
            return s->has_hidden_axis() ? 1 : s->len;
        });
}

static std::vector<std::size_t> compute_dims(const operation& op,
                                             const std::vector<std::size_t>& idims)
{
    shape s{shape::float_type, idims};
    return op.compute_shape({s}).lens();
}

MIGRAPHX_DEBUG_USED static std::vector<std::size_t>
compute_dims(const std::vector<operation>& ops, const std::vector<std::size_t>& idims)
{
    shape s{shape::float_type, idims};
    for(const auto& op : ops)
        s = op.compute_shape({s});
    return s.lens();
}

shape_transform_descriptor shape_transform_descriptor::create(const std::vector<std::size_t>& dims,
                                                              const std::vector<operation>& ops)
{
    shape_transform_descriptor result{dims};
    if(not result.apply(ops))
        return {};
    result.simplify();
    assert(compute_dims(ops, dims) == compute_dims(result.generate(), dims));
    return result;
}

static bool is_broadcast_only(const std::vector<dimension>& src_dims,
                              const std::vector<dimension>& dst_dims)
{
    return std::equal(src_dims.begin(),
                      src_dims.end(),
                      dst_dims.begin(),
                      dst_dims.end(),
                      [](const auto& src_dim, const auto& dst_dim) {
                          if(src_dim.subdimensions.size() != dst_dim.subdimensions.size())
                              return false;
                          auto match_sub_dim = [](const dimension::sub& src_sub,
                                                  const dimension::sub& dst_sub) {
                              if(src_sub.len == 1)
                                  return true;
                              return src_sub.len == dst_sub.len;
                          };
                          auto [src_it, dst_it] = std::mismatch(src_dim.subdimensions.begin(),
                                                                src_dim.subdimensions.end(),
                                                                dst_dim.subdimensions.begin(),
                                                                dst_dim.subdimensions.end(),
                                                                match_sub_dim);
                          if(src_it == src_dim.subdimensions.end())
                              return true;
                          // One mismatch is fine as long as the dimension is still the same size
                          if(src_dim.len() != dst_dim.len())
                              return false;
                          return std::equal(std::next(src_it),
                                            src_dim.subdimensions.end(),
                                            std::next(dst_it),
                                            dst_dim.subdimensions.end(),
                                            match_sub_dim);
                      });
}

template <class Dimensions, class Predicate>
static auto find_subdimension_with_dimension(Dimensions& dims, Predicate pred)
    -> std::pair<decltype(&dims[0]), decltype(&dims[0].subdimensions.front())>
{
    for(auto& dim : dims)
    {
        auto it = std::find_if(dim.subdimensions.begin(), dim.subdimensions.end(), pred);
        if(it != dim.subdimensions.end())
            return {&dim, &(*it)};
    }
    return {nullptr, nullptr};
}

// Class to handle axes rebase adjustment for ambiguous reshape transformations
//
// This class solves an ambiguity problem that arises when shape_transform_descriptor
// records reshape operations involving dimensions of size 1. When reshaping with 1 dims,
// there are multiple valid ways to split/assign axes, and the descriptor may not match
// the expected layout when rebasing.
//
// Why this is needed:
// When shape_transform_descriptor reshapes from [4, 1, 4] to [4, 1, 1, 4], it could record:
// - Option 1: [4:0], [1:1x0], [1:1x1], [4:2] (axis 1 split into two)
// - Option 2: [4:0], [1:1], [1:2,0], [4:2,1] (axis 2 split, with 1 inserted)
// Both are valid, but when rebasing, we need to adjust to match the expected dimensions.
// A similar issue occurs when squeezing 1 dims - the axis assignment becomes ambiguous.
//
// The adjustment process:
// - Identifies "shortage" axes (target dim > current subdimensions)
// - Finds "excess" axes or unassigned broadcast dimensions
// - Redistributes subdimensions to resolve the ambiguity
// - Uses hidden_axis to track broadcast dimensions for proper placement
struct rebase_ambiguity_resolver
{
    using axes_map_t = std::map<std::size_t, std::vector<dimension::sub*>>;

    // Structure to bundle axis-related information for cleaner parameter passing
    struct axis_info
    {
        std::size_t saxis;    // The shortage axis that needs subdimensions
        std::size_t excess;   // The amount of excess subdimensions available
        std::size_t base_dim; // The base dimension size
        std::size_t axis;     // The current axis being processed
    };

    rebase_ambiguity_resolver(shape_transform_descriptor& d, const std::vector<std::size_t>& ds)
        : desc(&d), dims(&ds)
    {
    }

    // Main entry point that orchestrates the axes adjustment process
    // Returns the axes mapping that can be used for rebase
    auto resolve()
    {
        std::vector<std::pair<dimension::sub, std::size_t>> subs_to_insert;
        {
            axes_map_t axes_map = group_axes(desc->dimensions);

            find_shortage_axes(axes_map);

            if(shortage_axes.empty())
                return axes_map;

            if(try_trivial_direct_mapping())
                return regroup_axes();

            process_axis_groups(axes_map, subs_to_insert);

            if(shortage_axes.size() == initial_shortage_count)
                return axes_map;
        }
        insert_moved_axes(subs_to_insert);

        swap_closer_axes();

        sort_hidden_axes_groups();
        sort_moved_axes_groups();

        return regroup_axes();
    }

    private:
    template <class T, class U>
    static auto check_div(T x, U y) -> decltype(x / y)
    {
        if(y == 0)
            return 0;
        if((x % y) != 0)
            return 0;
        return x / y;
    }

    axes_map_t regroup_axes()
    {
        axes_map_t result = group_axes(desc->dimensions);
        renumber_axes(result);
        return result;
    }

    // Identifies axes where the target dimension is larger than current subdimensions
    // These are "shortage" axes that need subdimensions due to ambiguous axis assignment
    void find_shortage_axes(const axes_map_t& axes_map)
    {
        for(const auto& [axis, subs] : axes_map)
        {
            assert(axis < dims->size());
            auto dim      = (*dims)[axis];
            auto shortage = check_div(dim, len(subs));
            if(shortage < 2)
                continue;
            shortage_axes.emplace(shortage, axis);
        }
        initial_shortage_count = shortage_axes.size();
    }

    bool try_trivial_direct_mapping()
    {
        if(desc->lens() != *dims)
            return false;
        if(not std::all_of(
               desc->dimensions.begin(), desc->dimensions.end(), [&](const dimension& d) {
                   if(d.subdimensions.empty())
                       return false;
                   if(d.len() == 1)
                       return true;
                   if(std::any_of(d.subdimensions.begin(),
                                  d.subdimensions.end(),
                                  [&](const dimension::sub& s) {
                                      if(s.origin_axis().empty())
                                          return false;
                                      if(s.origin_axis().size() != 1)
                                          return true;
                                      if(s.len == 1)
                                          return false;
                                      if(s.has_hidden_axis())
                                          return false;
                                      return ((*dims)[s.origin_axis().front()] != s.len);
                                  }))
                       return false;
                   if(d.subdimensions.size() == 1)
                       return true;
                   auto n1dims = std::count_if(d.subdimensions.begin(),
                                               d.subdimensions.end(),
                                               [](const dimension::sub& s) { return s.len == 1; });
                   return n1dims + 1 == d.subdimensions.size();
               }))
            return false;
        std::vector<std::size_t> axes;
        for_each_subdimension(desc->dimensions, [&](auto& s) {
            if(s.origin_axis().empty())
                return;
            axes.push_back(s.origin_axis().front());
        });
        // TODO: Handle permutations
        if(not std::is_sorted(axes.begin(), axes.end()))
            return false;
        for(std::size_t i : range(desc->dimensions.size()))
        {
            auto& dim = desc->dimensions[i];
            if(dim.subdimensions.empty())
                continue;
            auto sub = std::find_if(dim.subdimensions.begin(),
                                    dim.subdimensions.end(),
                                    [&](const dimension::sub& s) { return s.len != 1; });
            if(sub == dim.subdimensions.end())
                sub = dim.subdimensions.begin();
            sub->expose();
            sub->axis = {i};

            auto remove_axis = [](dimension::sub& s) {
                s.axis.clear();
                s.hidden_axis.clear();
                s.len = 1;
            };
            std::for_each(dim.subdimensions.begin(), sub, remove_axis);
            std::for_each(std::next(sub), dim.subdimensions.end(), remove_axis);
        }
        shortage_axes.clear();
        return true;
    }

    // Processes each axis group to resolve ambiguous axis assignments
    // This is the core logic that fixes mismatches from reshape ambiguity
    //
    // The process for each axis group:
    // 1. Calculate if there's excess (more subdimensions than needed)
    // 2. Find a matching shortage axis that needs exactly that excess
    // 3. Try to swap (for broadcast dimensions) or move subdimensions
    void process_axis_groups(const axes_map_t& axes_map,
                             std::vector<std::pair<dimension::sub, std::size_t>>& subs_to_insert)
    {
        for_each_axis_group(axes_map,
                            [&](std::size_t axis,
                                const std::vector<dimension::sub*>& subs,
                                std::size_t excess,
                                std::size_t base_dim) {
                                auto saxes = shortage_axes.equal_range(excess);
                                if(saxes.first == saxes.second)
                                    return;

                                auto saxis_it =
                                    find_nearest_shortage_axis(saxes.first, saxes.second, axis);

                                axis_info info{saxis_it->second, excess, base_dim, axis};

                                // Try to swap an axis
                                if(try_swap_axis(subs, info))
                                {
                                    shortage_axes.erase(saxis_it);
                                    return;
                                }

                                if(subs.size() != 1)
                                    return;

                                // Move the shortage to the excess dim
                                if(move_shortage_to_excess(subs, info, subs_to_insert))
                                {
                                    shortage_axes.erase(saxis_it);
                                }
                            });
    }

    // Helper that iterates over axis groups that have excess subdimensions
    // Calls the provided function for each axis with excess
    //
    // Two types of excess are handled:
    // 1. Regular axes with more subdimensions than needed
    // 2. "No-axis" subdimensions (from broadcasts) that can be assigned to any axis
    template <class F>
    void for_each_axis_group(const axes_map_t& axes_map, F f)
    {
        for(const auto& [axis, subs] : axes_map)
        {
            assert(axis < dims->size());
            auto dim    = (*dims)[axis];
            auto excess = check_div(len(subs), dim);
            if(excess < 2)
                continue;
            f(axis, subs, excess, dim);
        }
        // Look at dims with no axis
        for_each_subdimension(desc->dimensions, [&](auto& s) {
            if(not s.origin_axis().empty())
                return;
            auto excess = s.len;
            if(excess < 2)
                return;
            f(dims->size(), std::vector<dimension::sub*>{&s}, excess, 1);
        });
    }

    // Finds the shortage axis that is closest to the current axis
    // This helps maintain locality - we prefer to move subdimensions between nearby axes
    //
    // Example:
    // - Current axis: 2
    // - Shortage axes: {0->3, 1->4, 5->3} (axis->shortage pairs)
    // - Returns iterator to axis 1 (distance = |1-2| = 1)
    template <class Iterator>
    Iterator find_nearest_shortage_axis(Iterator first, Iterator last, std::size_t axis)
    {
        return std::min_element(first, last, by(std::less<>{}, [&](const auto& p) {
                                    std::int64_t a1 = p.second;
                                    std::int64_t a2 = axis;
                                    return std::abs(a1 - a2);
                                }));
    }

    // Attempts to reassign broadcast dimensions to resolve reshape ambiguity
    // Used when broadcast dims (originally dimension 1) need proper axis assignment
    //
    // This resolves cases where reshape with dimension 1 created ambiguous axis assignments
    bool try_swap_axis(const std::vector<dimension::sub*>& subs, const axis_info& info)
    {
        auto it = std::find_if(subs.begin(), subs.end(), [&](dimension::sub* sub) {
            if(not sub->has_hidden_axis() and not sub->origin_axis().empty())
                return false;
            return sub->len == info.excess;
        });
        if(it != subs.end())
        {
            auto* sub        = *it;
            sub->hidden_axis = {info.saxis, last_axis_split};
            return true;
        }
        return false;
    }

    // Moves subdimensions to resolve ambiguity from reshape operations with dimension 1
    // This physically relocates subdimensions when the ambiguous assignment needs correction.
    // The subdimension is pushed to subs_to_insert first because inserting here will
    // invalidate the references to the subdimensions.
    //
    // This fixes cases where reshape ambiguity left dimension 1s in wrong positions
    bool
    move_shortage_to_excess(const std::vector<dimension::sub*>& subs,
                            const axis_info& info,
                            std::vector<std::pair<dimension::sub, std::size_t>>& subs_to_insert)
    {
        auto dim_pair =
            find_subdimension_with_dimension(desc->dimensions, [&](const dimension::sub& s) {
                if(s.axis.size() != 1)
                    return false;
                return s.axis.front() == info.saxis;
            });
        if(dim_pair.first == nullptr)
            return false;
        auto* dim = dim_pair.first;
        auto* sub = dim_pair.second;
        assert(sub != nullptr);
        if(sub->len != 1)
            return false;
        if(dim->subdimensions.size() == 1)
            return false;
        subs_to_insert.push_back({dimension::sub{info.excess, {}, {info.saxis}}, info.axis});
        sub->axis.clear();
        subs.front()->len = info.base_dim;
        return true;
    }

    // Inserts the subdimensions that were marked for movement
    // These subdimensions are inserted at their new positions
    //
    // Example:
    // - Sub [3:,{1}] needs to be inserted at position after axis 0
    // - Finds the dimension containing axis 0 subdimensions
    // - Inserts after the last subdimension with axis 0
    void
    insert_moved_axes(const std::vector<std::pair<dimension::sub, std::size_t>>& subs_to_insert)
    {
        for(const auto& [sub, pos_axis] : subs_to_insert)
        {
            // Inline insert_single_axis
            auto equal_to_pos_axis = [&, lpos_axis = pos_axis](const dimension::sub& s) {
                if(s.origin_axis().empty())
                    return false;
                return s.origin_axis().front() == lpos_axis;
            };
            auto dim_pair = find_subdimension_with_dimension(desc->dimensions, equal_to_pos_axis);
            assert(dim_pair.first != nullptr);
            auto* dim = dim_pair.first;
            auto it   = std::find_if(
                dim->subdimensions.begin(), dim->subdimensions.end(), equal_to_pos_axis);
            assert(it != dim->subdimensions.end());
            dim->subdimensions.insert(std::next(it), sub);
            moved_axes.insert(sub.origin_axis().front());
        }
    }

    static bool has_hidden_axis(const dimension::sub* s) { return s->has_hidden_axis(); }

    static const std::vector<std::size_t>& get_hidden_axis(const dimension::sub* s)
    {
        return s->hidden_axis;
    }

    // Optimizes the placement of hidden axes to group related dimensions
    // This helps clean up after resolving reshape ambiguities with dimension 1s
    //
    // This ensures the final axis assignment is clean and predictable
    void swap_closer_axes()
    {
        auto subs = get_pointer_subdimensions(desc->dimensions);

        group_find(subs.begin(), subs.end(), &has_hidden_axis, [](auto start, auto last) {
            if(std::distance(start, last) < 2)
                return;

            adjacent_for_each(start, last, [&](dimension::sub* s1, dimension::sub* s2) {
                if(s1->hidden_axis.empty())
                    return;
                if(s2->hidden_axis.empty())
                    return;
                assert(s1->axis.empty());
                assert(s2->axis.empty());

                auto it = min_element_if(
                    start,
                    last,
                    [&](const dimension::sub* s) {
                        // Valid if same len as s2 and hidden_axis greater than s1
                        return s->len == s2->len and s->hidden_axis > s1->hidden_axis;
                    },
                    by(std::less<>{}, &get_hidden_axis));

                if(it == last)
                    return;

                auto* next_sub = *it;
                assert(s1->hidden_axis < next_sub->hidden_axis);
                if(next_sub->hidden_axis.empty())
                    return;
                if(next_sub == s2)
                    return;
                if(next_sub->len != s2->len)
                    return;
                std::swap(*s2, *next_sub);
            });
        });
    }

    static auto get_hidden_axis_group(const dimension::sub* s)
    {
        if(s->hidden_axis.empty())
            return std::numeric_limits<std::size_t>::max();
        return s->hidden_axis.front();
    }

    template <class Pred>
    static auto sort_group_if(Pred pred)
    {
        return [=](auto start, auto last) {
            if(std::distance(start, last) < 2)
                return;
            if(not pred(*start))
                return;
            auto r    = range(start, last);
            auto axes = views::transform(r, [](MIGRAPHX_CPPCHECK_CONST dimension::sub* s) -> auto& {
                return s->hidden_axis;
            });
            std::sort(axes.begin(), axes.end());
        };
    }

    auto get_is_moved_axis()
    {
        return [this](const dimension::sub* s) {
            if(s->origin_axis().empty())
                return false;
            return contains(moved_axes, s->origin_axis().front());
        };
    }

    // Sorts groups of hidden axes to to reduce transposition.
    void sort_hidden_axes_groups()
    {
        auto subs = get_pointer_subdimensions(desc->dimensions);
        group_unique(subs.begin(),
                     subs.end(),
                     sort_group_if([](dimension::sub* s) { return not s->hidden_axis.empty(); }),
                     by(std::equal_to<>{}, &get_hidden_axis_group));
    }

    // If subdimensions are moved together then sort to reduce transposition.
    void sort_moved_axes_groups()
    {
        for(auto& d : desc->dimensions)
        {
            auto asubs = views::transform(d.subdimensions, [](dimension::sub& s) { return &s; });
            group_unique(asubs.begin(),
                         asubs.end(),
                         sort_group_if(get_is_moved_axis()),
                         by(std::equal_to<>{}, get_is_moved_axis()));
        }
    }

    private:
    shape_transform_descriptor* desc;
    const std::vector<std::size_t>* dims;
    std::multimap<std::size_t, std::size_t> shortage_axes;
    std::size_t initial_shortage_count = 0;
    std::unordered_set<std::size_t> moved_axes;

    static const std::size_t last_axis_split = std::numeric_limits<std::size_t>::max();
};

shape_transform_descriptor shape_transform_descriptor::rebase(const std::vector<std::size_t>& dims,
                                                              bool broadcast) const
{
    auto result   = *this;
    auto axes_map = rebase_ambiguity_resolver{result, dims}.resolve();
    for(auto& [axis, subs] : axes_map)
    {
        assert(axis < dims.size());
        auto dim       = dims[axis];
        if(dim == len(subs))
        {
            if(not broadcast)
            {
                for(auto* sub : subs)
                    sub->expose();
            }
        }
        else if(dim == 1)
        {
            for(auto* sub : subs)
            {
                if(not sub->has_hidden_axis())
                    sub->len = 1;
            }
        }
        else if(subs.size() == 1)
        {
            subs.front()->len = dim;
            if(broadcast)
                subs.front()->hide();
            else
                subs.front()->expose();
        }
        else if(dim == visible_len(subs))
        {
            for(auto* sub : subs)
            {
                if(sub->has_hidden_axis())
                {
                    sub->expose();
                    sub->len = 1;
                }
            }
        }
        else
            return {};
    }
    for(auto& dim : result.dimensions)
        remove_empty_sub_dims(dim.subdimensions);
    if(broadcast and not is_broadcast_only(dimensions, result.dimensions))
        return {};

    return result;
}
static dimension::sub* get_last_subdimension(std::vector<dimension>& dims)
{
    if(dims.empty())
        return {};
    auto& d = dims.back();
    if(d.subdimensions.empty())
        return nullptr;
    return &d.subdimensions.back();
}

bool shape_transform_descriptor::apply(const std::vector<operation>& ops)
{
    std::vector<std::size_t> dims;
    std::transform(dimensions.begin(),
                   dimensions.end(),
                   std::back_inserter(dims),
                   [](const dimension& d) { return d.len(); });
    for(const auto& op : ops)
    {
        auto v = op.to_value();
        if(contains({"reshape", "squeeze", "unsqueeze", "flatten"}, op.name()))
        {
            dims = compute_dims(op, dims);
            if(not apply_reshape(dims))
                return false;
        }
        else if(op.name() == "transpose")
        {
            dims = compute_dims(op, dims);
            if(not apply_transpose(v["permutation"].to_vector<std::int64_t>()))
                return false;
        }
        else if(op.name() == "multibroadcast")
        {
            dims = compute_dims(op, dims);
            // cppcheck-suppress knownConditionTrueFalse
            if(not apply_broadcast(dims))
                return false;
        }
        else if(op.name() == "broadcast")
        {
            dims = compute_dims(op, dims);
            // cppcheck-suppress knownConditionTrueFalse
            if(not apply_broadcast(dims, v["axis"].to<std::size_t>()))
                return false;
        }
        else if(op.name() != "contiguous")
        {
            return false;
        }
    }
    return true;
}
bool shape_transform_descriptor::apply_reshape(const std::vector<std::size_t>& rdims)
{
    std::vector<std::size_t> idims;
    transform(get_all_subdimensions(dimensions),
              std::back_inserter(idims),
              std::mem_fn(&dimension::sub::len));
    auto cdims = common_dims::compute(idims, rdims).dims;
    if(not cdims.empty() and not apply_reshape_impl(cdims))
        return false;
    return apply_reshape_impl(rdims);
}
bool shape_transform_descriptor::apply_reshape_impl(const std::vector<std::size_t>& rdims)
{
    assert(migraphx::elements(rdims) == this->elements());
    if(migraphx::equal(
           dimensions, rdims, [](const dimension& d, std::size_t rdim) { return d.len() == rdim; }))
        return true;
    std::vector<dimension> new_dims;
    auto subs     = get_all_subdimensions(dimensions);
    std::size_t i = 0;
    std::size_t r = 0;
    while(i < subs.size() and r < rdims.size())
    {
        const auto& sub = subs[i];
        auto idim       = sub.len;
        auto rdim       = rdims[r];
        if(idim == rdim)
        {
            new_dims.push_back({{sub}});
        }
        // squeeze
        else if(rdim > idim)
        {
            auto start = subs.begin() + i;
            auto it = compute_end_dim(start, subs.end(), rdim, std::mem_fn(&dimension::sub::len));
            if(it == start)
                return false;
            assert(it != subs.end());
            auto n = it - start;
            i += n;
            new_dims.push_back({{start, it + 1}});
        }
        // unsqueeze
        else // if(rdim < idim)
        {
            auto start = rdims.begin() + r;
            auto it    = compute_end_dim(start, rdims.end(), idim, id{});
            if(it == start)
                return false;
            assert(it != rdims.end());
            auto n = it - start;
            r += n;
            transform(range(n + 1), std::back_inserter(new_dims), [&](auto j) -> dimension {
                auto new_sub = sub;
                new_sub.add_split_axis(j);
                new_sub.len = start[j];
                return {{new_sub}};
            });
        }
        r++;
        i++;
    }

    // Handle trailing 1s
    if(new_dims.size() < rdims.size() and not new_dims.empty())
    {
        auto* sub          = get_last_subdimension(new_dims);
        auto axis          = sub == nullptr ? std::vector<std::size_t>{} : sub->axis;
        auto trailing_dims = range(rdims.begin() + new_dims.size(), rdims.end());
        if(any_of(trailing_dims, [](auto d) { return d != 1; }))
            return false;
        if(distance(trailing_dims) > 1)
            sub->add_split_axis(0);
        transform(range(distance(trailing_dims)),
                  std::back_inserter(new_dims),
                  [&](std::size_t j) -> dimension {
                      dimension::sub s{1, axis};
                      s.add_split_axis(j + 1);
                      return {{s}};
                  });
    }
    assert(rdims.size() == new_dims.size());
    if(rdims.size() != new_dims.size())
        return false;
    dimensions = new_dims;
    return true;
}

bool shape_transform_descriptor::apply_transpose(const std::vector<std::int64_t>& permutation)
{
    if(permutation.size() != dimensions.size())
        return false;
    dimensions = reorder_dims(dimensions, permutation);
    return true;
}

bool shape_transform_descriptor::apply_broadcast(const std::vector<std::size_t>& out_lens,
                                                 optional<std::size_t> axis)
{
    auto offset = axis.value_or(out_lens.size() - dimensions.size());

    std::vector<dimension> new_dims;
    std::transform(out_lens.begin(),
                   out_lens.begin() + offset,
                   std::back_inserter(new_dims),
                   [&](auto len) -> dimension {
                       return {{dimension::sub{len, {}}}};
                   });
    std::transform(dimensions.begin(),
                   dimensions.end(),
                   out_lens.begin() + offset,
                   std::back_inserter(new_dims),
                   [&](const dimension& dim, auto len) -> dimension {
                       if(len == dim.len())
                           return dim;
                       if(dim.len() != 1)
                           MIGRAPHX_THROW("Wrong out_lens for broadcast");
                       auto new_subs = dim.subdimensions;
                       if(not new_subs.empty())
                       {
                           new_subs.front().len = len;
                       }
                       for(auto& s : new_subs)
                       {
                           s.hide();
                       }
                       return {new_subs};
                   });
    std::transform(out_lens.begin() + offset + dimensions.size(),
                   out_lens.end(),
                   std::back_inserter(new_dims),
                   [&](auto len) -> dimension {
                       return {{dimension::sub{len, {}}}};
                   });
    assert(out_lens.size() == new_dims.size());
    dimensions = new_dims;
    return true;
}

// Remove subdimensions of 1
static void remove_1_sub_dims(std::vector<dimension::sub>& subdimensions)
{
    subdimensions.erase(std::remove_if(subdimensions.begin(),
                                       subdimensions.end(),
                                       [&](const dimension::sub& d) { return d.len == 1; }),
                        subdimensions.end());
    if(subdimensions.empty())
        subdimensions.push_back({1, {}});
}

void dimension::simplify()
{
    if(subdimensions.size() < 2)
        return;
    remove_1_sub_dims(subdimensions);
    // Flatten adjacent dimensions
    adjacent_for_each(subdimensions.begin(), subdimensions.end(), [&](sub& d1, sub& d2) {
        if(d1.origin_axis().size() < 2)
            return;
        if(d2.origin_axis().size() < 2)
            return;
        if(d1.has_hidden_axis() != d2.has_hidden_axis())
            return;
        if(not std::equal(d1.origin_axis().begin(),
                          d1.origin_axis().end() - 1,
                          d2.origin_axis().begin(),
                          d2.origin_axis().end() - 1))
            return;
        auto a1 = d1.origin_axis().back();
        auto a2 = d2.origin_axis().back();
        assert(a2 != a1);
        if(a2 <= a1)
            return;
        if((a2 - a1) != 1)
            return;
        d2.len = d1.len * d2.len;
        d1.len = 1;
    });
    remove_1_sub_dims(subdimensions);
}

// Search all subdimensions and return the subdimensions vector, an iterator
// to the subdimension found and an optional iterator to the previous
// subdimension if available.
template <class Predicate>
static auto find_subdimension_with_prev(shape_transform_descriptor& td, Predicate p)
{
    dimension* prev_dim = nullptr;
    for(auto& d : td.dimensions)
    {
        auto it = std::find_if(d.subdimensions.begin(), d.subdimensions.end(), p);
        if(it != d.subdimensions.end())
        {
            decltype(std::make_optional(it)) prev = nullopt;
            if(it == d.subdimensions.begin())
            {
                if(prev_dim != nullptr and not prev_dim->subdimensions.empty())
                {
                    prev = std::prev(prev_dim->subdimensions.end());
                }
            }
            else
            {
                prev = std::prev(it);
            }
            return std::make_tuple(&d.subdimensions, it, prev);
        }
        prev_dim = &d;
    }
    MIGRAPHX_THROW("Searching for non-existent subdimension");
}

static bool is_broadcast_dim(const dimension& d)
{
    if(d.len() == 1)
        return false;
    assert(not d.subdimensions.empty());
    if(d.subdimensions.size() != 1)
        return false;
    const auto& sub = d.subdimensions.front();
    return sub.axis.empty();
}

static bool missing_leading_axis(const dimension& d)
{
    if(d.subdimensions.empty())
        return true;
    const auto& sub = d.subdimensions.front();
    return sub.origin_axis().empty();
}

static void set_broadcast_dim(dimension& d, std::size_t axis)
{
    if(d.subdimensions.empty())
        d.subdimensions.push_back({1, {axis}});
    else
    {
        assert(d.subdimensions.front().hidden_axis.empty());
        d.subdimensions.front().hidden_axis = {axis};
    }
}

// If an axis is split and some dimensions are hidden and others are not, then
// remove the hidden axis so only the non-hidden axis is used in
// simplificaiton
static void remove_split_hidden_axes(std::map<std::size_t, std::vector<dimension::sub*>>& axes_map)
{
    for(auto&& p : axes_map)
    {
        auto& subs = p.second;
        if(std::all_of(subs.begin(), subs.end(), [](const dimension::sub* s) {
               return s->has_hidden_axis();
           }))
            continue;
        for(auto* sub : subs)
        {
            if(not sub->has_hidden_axis())
                continue;
            sub->hidden_axis.clear();
        }
        // Remove the subdimesions that no longer have an axis
        subs.erase(std::remove_if(subs.begin(),
                                  subs.end(),
                                  [](const dimension::sub* s) {
                                      return s->axis.empty() and s->hidden_axis.empty();
                                  }),
                   subs.end());
    }
    // Remove axis from group if empty
    erase_if(axes_map, [](auto&& p) { return p.second.empty(); });
}

// Replace the hidden axis that is split with an axis that is missing
static void fill_split_hidden_axes(std::map<std::size_t, std::vector<dimension::sub*>>& axes_map,
                                   const std::vector<dimension>& dimensions,
                                   std::size_t rank)
{
    // Create a reverse map of the subdimensions to the position
    std::unordered_map<const dimension::sub*, std::size_t> sub_pos_map;
    for_each_subdimension(dimensions,
                          range(std::numeric_limits<std::size_t>::max()),
                          [&](const dimension::sub& sub, std::size_t i) { sub_pos_map[&sub] = i; });
    for(auto&& p : axes_map)
    {
        auto axis = p.first;
        auto subs = p.second;
        if(subs.size() < 2)
            continue;
        std::sort(subs.begin(), subs.end(), by(std::less<>{}, [](const dimension::sub* s) {
                      return s->origin_axis();
                  }));
        if(not std::all_of(subs.begin(), subs.end(), [](const dimension::sub* s) {
               return s->has_hidden_axis();
           }))
            continue;

        auto it = std::adjacent_find(
            subs.begin(), subs.end(), [&](const dimension::sub* s1, const dimension::sub* s2) {
                return sub_pos_map.at(s1) + 1 != sub_pos_map.at(s2);
            });
        if(it != subs.end())
            continue;
        auto needed_axes  = range(axis + 1, axis + subs.size());
        auto missing_axes = reverse(range(needed_axes.begin(), find_if(needed_axes, [&](auto a) {
                                              if(a >= rank)
                                                  return true;
                                              return contains(axes_map, a);
                                          })));
        for_each(missing_axes.begin(),
                 missing_axes.end(),
                 subs.rbegin(),
                 [&](std::size_t axis, dimension::sub* sub) {
                     sub->hidden_axis = {axis};
                     axes_map[axis].push_back(sub);
                 });
        // Remove the subdimansions that have a different axis
        auto& orig_subs = p.second;
        orig_subs.erase(std::remove_if(orig_subs.begin(),
                                       orig_subs.end(),
                                       [&](const dimension::sub* s) {
                                           return s->origin_axis().front() != axis;
                                       }),
                        orig_subs.end());
    }
}

// If this is scalar, then remove all axes
static void remove_scalar_axis(std::vector<dimension>& dimensions)
{
    dimension::sub* s = nullptr;
    for(auto& d : dimensions)
    {
        auto has_axis = [](const dimension::sub& x) { return not x.origin_axis().empty(); };
        auto it       = std::find_if(d.subdimensions.begin(), d.subdimensions.end(), has_axis);
        if(it == d.subdimensions.end())
            continue;
        if(s != nullptr)
            return;
        if(std::count_if(std::next(it), d.subdimensions.end(), has_axis) > 0)
            return;
        s = &*it;
    }
    if(s != nullptr)
    {
        if(s->has_hidden_axis())
            s->hidden_axis.clear();
        if(s->len == 1)
            s->axis.clear();
    }
}

static void collapse_1_dims(std::vector<dimension>& dimensions)
{
    // Find a dimension that ends with a subdimension of 1 with a single axis,
    // and is followed by subdimension in the next dimension of 1 that has a
    // split axis. It will remove the trailing subdimension and update the
    // leading subdimension to use the axis from the trailing subdimension.
    adjacent_for_each(dimensions.begin(), dimensions.end(), [&](dimension& d1, dimension& d2) {
        if(d1.subdimensions.size() < 2)
            return;
        if(d2.subdimensions.empty())
            return;
        if(d2.len() != 1)
            return;
        const auto& sub1 = d1.subdimensions.back();
        auto& sub2       = d2.subdimensions.front();
        if(sub1.axis.size() != 1)
            return;
        if(sub2.axis.size() < 2)
            return;
        if(sub1.len != 1)
            return;
        if(sub2.len != 1)
            return;
        sub2.axis = sub1.axis;
        d1.subdimensions.pop_back();
    });

    renumber_axes(dimensions);
}

static void insert_empty_1s(std::vector<dimension>& dimensions, std::size_t rank)
{
    if(dimensions.empty())
        return;
    transform(dimensions,
              range(rank),
              dimensions.begin(),
              [](const dimension& d, std::size_t i) -> dimension {
                  auto result = dimension::sub{d.len(), {i}};
                  if(result.len > 1)
                      result.hide();
                  return {{result}};
              });
    if(rank > dimensions.size())
    {
        transform(range(dimensions.size(), rank),
                  std::back_inserter(dimensions.back().subdimensions),
                  [](std::size_t i) { return dimension::sub{1, {i}}; });
    }
}

// Find missing axes. This will store a mapping between the missing
// axis and the next available axis.
static std::map<std::size_t, std::size_t>
find_missing_axes(const std::map<std::size_t, std::vector<dimension::sub*>>& axes_map,
                  std::size_t rank)
{
    std::map<std::size_t, std::size_t> missing_axes;
    for(auto axis : range(rank))
    {
        if(contains(axes_map, axis))
            continue;
        auto it            = axes_map.upper_bound(axis);
        missing_axes[axis] = it == axes_map.end() ? rank : it->first;
    }
    return missing_axes;
}

// Find broadcasted dimensions. This will store a map from the next axis
// to the indices of the previous dimensions that are being broadcasted.
static std::map<std::size_t, std::deque<std::size_t>>
find_broadcasted_dims(const std::vector<dimension>& dimensions, std::size_t rank)
{
    std::map<std::size_t, std::deque<std::size_t>> broadcast_dims_map;
    group_find(
        dimensions.begin(), dimensions.end(), &missing_leading_axis, [&](auto start, auto last) {
            auto axis = rank;
            if(last != dimensions.end())
            {
                assert(not last->subdimensions.empty());
                const auto& sub = last->subdimensions.front();
                assert(not sub.origin_axis().empty());
                axis = sub.origin_axis().front();
            }
            std::deque<std::size_t> dims(std::distance(start, last));
            std::iota(dims.begin(), dims.end(), std::distance(dimensions.begin(), start));
            broadcast_dims_map[axis] = dims;
        });
    return broadcast_dims_map;
}

void shape_transform_descriptor::simplify()
{
    for(auto& d : dimensions)
        d.simplify();

    remove_scalar_axis(dimensions);

    std::map<std::size_t, std::size_t> missing_axes;
    std::vector<std::size_t> last_axis;
    {
        // Group axes
        auto axes_map = group_axes(dimensions);
        if(axes_map.empty())
        {
            insert_empty_1s(dimensions, rank);
            return;
        }

        remove_split_hidden_axes(axes_map);
        fill_split_hidden_axes(axes_map, dimensions, rank);
        renumber_axes(axes_map);

        // Find last axis
        last_axis = std::prev(axes_map.end())->second.back()->origin_axis();

        missing_axes = find_missing_axes(axes_map, rank);
    }

    std::map<std::size_t, std::deque<std::size_t>> broadcast_dims_map =
        find_broadcasted_dims(dimensions, rank);

    // Reinsert removed axis of 1. This tries to insert the missing axis next
    // to an adjacent axis or used as one of the broadcasted axes in order to
    // minimize transposition.
    for(auto&& p : missing_axes)
    {
        auto missing_axis = p.first;
        auto next_axis    = p.second;
        auto missing_sub  = dimension::sub{1, {missing_axis}};
        // If next_axis is the rank that means there isnt another axis to
        // search for, so instead try to insert the axis at the end.
        if(next_axis == rank)
        {
            auto [sub, it, prev] = find_subdimension_with_prev(
                *this, [&](const dimension::sub& s) { return s.origin_axis() == last_axis; });
            // Check if we can insert it at the end
            auto bdims = broadcast_dims_map.find(rank);
            if(bdims != broadcast_dims_map.end() and not bdims->second.empty())
            {
                auto bdim = bdims->second.front();
                bdims->second.pop_front();
                set_broadcast_dim(dimensions[bdim], missing_axis);
            }
            else
            {
                auto next = std::find_if(std::next(it), sub->end(), [&](const dimension::sub& s) {
                    if(s.len != 1)
                        return true;
                    if(s.axis.empty())
                        return true;
                    return s.axis.front() > missing_axis;
                });
                sub->insert(next, missing_sub);
            }
            // Update last_axis if this is inserted afterwards
            if(missing_axis > last_axis.front())
                last_axis = {missing_axis};
        }
        else
        {
            // Search for the subdimension that has the next axis and try to
            // insert the axis before it will be in order.
            auto [sub, it, prev] = find_subdimension_with_prev(*this, [&](const dimension::sub& s) {
                if(s.origin_axis().empty())
                    return false;
                if(s.origin_axis().front() != next_axis)
                    return false;
                if(s.origin_axis().size() == 1)
                    return true;
                assert(s.origin_axis().size() == 2);
                return s.origin_axis().back() == 0;
            });
            bool in_order        = false;
            if(prev.has_value() and not(*prev)->origin_axis().empty())
                in_order = (*prev)->origin_axis().front() == missing_axis - 1;
            // If the axis is not inorder then see if we can find a broadcast axis to place it
            auto bdims =
                in_order ? broadcast_dims_map.end() : broadcast_dims_map.upper_bound(missing_axis);
            if(bdims != broadcast_dims_map.end() and not bdims->second.empty())
            {
                auto bdim = bdims->second.front();
                bdims->second.pop_front();
                set_broadcast_dim(dimensions[bdim], missing_axis);
            }
            else
            {
                sub->insert(it, missing_sub);
            }
        }
    }

    collapse_1_dims(dimensions);
}

static std::size_t get_len(const dimension::sub& s, const std::vector<std::size_t>& input_dims)
{
    if(input_dims.empty())
        return s.len;
    if(s.axis.empty())
        return s.len;
    auto dim = input_dims.at(s.axis.front());
    if(dim == 0)
        return s.len;
    if(dim == 1)
        return 1;
    if(s.axis.size() == 1)
        return dim;
    return s.len;
}

static operation make_reshape_squeeze(const std::vector<dimension>& new_dims)
{
    // Can use squeeze
    if(std::all_of(new_dims.begin(), new_dims.end(), [](const dimension& d) {
           if(d.subdimensions.size() < 2)
               return true;
           auto n = std::count_if(d.subdimensions.begin(),
                                  d.subdimensions.end(),
                                  [&](const dimension::sub& s) { return s.len == 1; });
           return n >= (d.subdimensions.size() - 1);
       }))
    {
        std::vector<std::size_t> base_axes = {0};
        transform_partial_sum(
            new_dims.begin(),
            std::prev(new_dims.end()),
            std::back_inserter(base_axes),
            std::plus<>{},
            [](const dimension& d) { return std::max<std::size_t>(1, d.subdimensions.size()); });
        auto get_squeezed_axes = [](const dimension& d, std::size_t base_axis) {
            std::vector<std::size_t> result;
            if(d.subdimensions.size() < 2)
                return result;
            auto idx = range(d.subdimensions.size());
            transform_if(
                idx.begin(),
                idx.end(),
                std::back_inserter(result),
                [&](std::size_t i) { return d.subdimensions[i].len == 1; },
                [&](std::size_t i) { return base_axis + i; });
            if(result.size() == d.subdimensions.size())
                result.pop_back();
            return result;
        };
        std::vector<std::size_t> axes;
        std::transform(new_dims.begin(),
                       new_dims.end(),
                       base_axes.begin(),
                       join_back_inserter(axes),
                       get_squeezed_axes);
        return make_op("squeeze", {{"axes", axes}});
    }
    else
    {
        std::vector<std::size_t> dims;
        std::transform(new_dims.begin(),
                       new_dims.end(),
                       std::back_inserter(dims),
                       [](const dimension& d) -> std::size_t {
                           if(is_broadcast_dim(d))
                               return 1;
                           return d.len();
                       });
        return make_op("reshape", {{"dims", dims}});
    }
}

static void flatten_broadcasted_dim(dimension::sub& s)
{
    if(s.axis.empty())
    {
        s.len = 1;
        s.expose();
    }
}

static operation make_reshape_unsqueeze(const std::vector<dimension::sub>& subs)
{
    bool use_reshape = false;
    std::unordered_set<std::size_t> all_1s;
    // Check if split dimensions are all additional 1s
    if(std::any_of(
           subs.begin(), subs.end(), [](const dimension::sub& s) { return s.axis.size() > 1; }))
    {
        auto subs2   = subs;
        auto by_axis = by(std::equal_to<>{}, [](const dimension::sub& s) -> int64_t {
            if(s.axis.empty())
                return -1;
            return s.axis.front();
        });
        group_by(
            subs2.begin(),
            subs2.end(),
            [&](auto start, auto last) {
                if(use_reshape)
                    return;
                // Number of elements
                auto n = std::distance(start, last);
                if(n < 2)
                    return;
                // Number of elements that are 1
                auto n1 =
                    std::count_if(start, last, [&](const dimension::sub& s) { return s.len == 1; });
                if(n == n1 and not start->axis.empty())
                    all_1s.insert(start->axis.front());
                use_reshape |= std::max<int64_t>(0, n - n1 - 1) > 0;
            },
            by_axis);
    }
    if(use_reshape)
    {
        std::vector<std::size_t> dims;
        std::transform(subs.begin(),
                       subs.end(),
                       std::back_inserter(dims),
                       [&](const dimension::sub& s) -> std::size_t {
                           if(s.axis.empty())
                               return 1;
                           return s.len;
                       });
        return make_op("reshape", {{"dims", dims}});
    }
    else
    {
        std::vector<std::size_t> axes;
        for(auto i : range(subs.size()))
        {
            const auto& sub = subs[i];
            if(sub.axis.size() == 1)
                continue;
            if(sub.len != 1 and not sub.axis.empty())
                continue;
            if(not sub.axis.empty() and contains(all_1s, sub.axis.front()) and sub.axis.back() == 0)
                continue;
            axes.push_back(i);
        }
        return make_op("unsqueeze", {{"axes", axes}});
    }
}

namespace {
struct operation_list
{
    std::vector<operation> ops;

    void push_back(const operation& op) { ops.push_back(op); }

    std::vector<operation> to_vector() &&
    {
        std::reverse(ops.begin(), ops.end());
        return std::move(ops);
    }
};

} // namespace

static bool has_no_axes(const dimension& d)
{
    return std::all_of(d.subdimensions.begin(), d.subdimensions.end(), [](const dimension::sub& s) {
        return s.axis.empty() and s.hidden_axis.empty();
    });
}
static bool has_axes(const dimension& d)
{
    return std::any_of(d.subdimensions.begin(), d.subdimensions.end(), [](const dimension::sub& s) {
        return not s.axis.empty();
    });
}

static std::vector<dimension::sub> attach_empty_axis(std::vector<dimension::sub> tsubs)
{
    // Inject additonal axis to compute transpose permutation better
    auto is_empty_axis = [](const auto& s) { return s.axis.empty(); };
    group_find(tsubs.begin(), tsubs.end(), is_empty_axis, [&](auto start, auto last) {
        if(start == tsubs.begin())
            return;
        auto base = std::prev(start);
        auto axis = base->axis;
        axis.push_back(0);
        std::for_each(start, last, [&](auto& s) {
            s.axis = axis;
            axis.back()++;
        });
    });
    return tsubs;
}

static std::vector<int64_t> find_permutation(const std::vector<dimension::sub>& subs)
{
    auto compare_sub = [](auto f) {
        return by(f, [](const dimension::sub& s) -> const auto& { return s.axis; });
    };
    return sort_permutation(subs, compare_sub(std::less<>{}));
}

// This will generate the operators to apply the shape transformation that is
// represented by this class. This is the order of operators that will be
// generated if needed:
//
// 1. Reshape/unsqueeze
// 2. Transpose
// 3. Broadcast
// 4. Reshape/squeeze
// 5. Broadcast
//
// This will generate operators backwards starting at 5 and going up. Steps 1-3
// are generated from the subdimensions and steps 4-5 are generated with the
// dimensions.
std::vector<operation>
shape_transform_descriptor::generate(const std::vector<std::size_t>& input_dims,
                                     bool no_broadcast) const
{
    operation_list result;
    std::vector<dimension> new_dims =
        input_dims.empty() ? dimensions : this->rebase(input_dims).dimensions;
    assert(input_dims.empty() or not new_dims.empty());
    if(no_broadcast)
    {
        for_each_subdimension(new_dims, &flatten_broadcasted_dim);
    }
    else
    {
        // Need broadcast
        if(std::any_of(new_dims.begin(), new_dims.end(), &is_broadcast_dim))
        {
            std::vector<std::size_t> out_lens;
            std::transform(new_dims.begin(),
                           new_dims.end(),
                           std::back_inserter(out_lens),
                           [](const dimension& d) { return d.len(); });
            auto startb     = std::find_if_not(new_dims.begin(), new_dims.end(), &has_no_axes);
            auto trailb     = std::find_if_not(startb, new_dims.end(), &has_axes);
            auto axis       = std::distance(new_dims.begin(), startb);
            auto extra_dims = axis + std::distance(trailb, new_dims.end());
            // Use broadcast instead of multibroadcast
            if(std::all_of(trailb, new_dims.end(), &has_no_axes) and extra_dims > 0 and
               axis < new_dims.size())
            {
                result.push_back(make_op("broadcast", {{"axis", axis}, {"out_lens", out_lens}}));
                new_dims.erase(trailb, new_dims.end());
                new_dims.erase(new_dims.begin(), new_dims.begin() + axis);
            }
            else
            {
                result.push_back(make_op("multibroadcast", {{"out_lens", out_lens}}));
            }
        }
        // If all the dimensions have no axes then there isnt anthing else to do
        // so just clear the new_dims
        if(std::all_of(new_dims.begin(), new_dims.end(), &has_no_axes))
            new_dims.clear();
        // Flatten broadcasted dimensions
        for(auto& d : new_dims)
        {
            if(d.subdimensions.size() != 1)
                continue;
            flatten_broadcasted_dim(d.subdimensions.front());
        }
    }
    // Need squeeze reshape
    if(std::any_of(new_dims.begin(), new_dims.end(), [](const dimension& d) {
           if(d.subdimensions.size() != 1)
               return true;
           return is_broadcast_dim(d);
       }))
    {
        result.push_back(make_reshape_squeeze(new_dims));
    }

    auto subs = get_all_subdimensions(new_dims);
    // Need multibroadcast
    if(std::any_of(subs.begin(), subs.end(), [&](const dimension::sub& s) {
           return s.axis.empty() and s.len != 1;
       }))
    {
        std::vector<std::size_t> out_lens;
        std::transform(subs.begin(),
                       subs.end(),
                       std::back_inserter(out_lens),
                       [&](const dimension::sub& s) { return s.len; });
        result.push_back(make_op("multibroadcast", {{"out_lens", out_lens}}));
    }

    // Flatten broadcasted subdimensions
    std::for_each(subs.begin(), subs.end(), &flatten_broadcasted_dim);

    auto permutation = find_permutation(attach_empty_axis(subs));
    // Need transpose
    if(not std::is_sorted(permutation.begin(), permutation.end()))
    {
        result.push_back(make_op("transpose", {{"permutation", invert_permutation(permutation)}}));
        subs = reorder_dims(subs, permutation);
    }
    // Need reshape unsqueeze
    if(std::any_of(
           subs.begin(), subs.end(), [](const dimension::sub& s) { return s.axis.size() != 1; }))
    {
        result.push_back(make_reshape_unsqueeze(subs));
    }
    return std::move(result).to_vector();
}

std::set<std::size_t> shape_transform_descriptor::find_broadcasted_axes() const
{
    std::set<std::size_t> result;
    for_each_subdimension(dimensions, [&](const dimension::sub& s) {
        if(s.has_hidden_axis())
            result.insert(s.hidden_axis.front());
    });
    return result;
}

bool shape_transform_descriptor::has_broadcast() const
{
    return std::any_of(dimensions.begin(), dimensions.end(), [&](const dimension& d) {
        return std::any_of(d.subdimensions.begin(),
                           d.subdimensions.end(),
                           [&](const dimension::sub& s) { return s.axis.empty() and s.len != 1; });
    });
}
void shape_transform_descriptor::flatten_broadcast()
{
    for(auto& d : dimensions)
        std::for_each(d.subdimensions.begin(), d.subdimensions.end(), &flatten_broadcasted_dim);
}

shape_transform_descriptor shape_transform_descriptor::to_common_from_src() const
{
    shape_transform_descriptor result;
    auto subs = get_all_subdimensions(this->dimensions);
    std::transform(subs.begin(),
                   subs.end(),
                   std::back_inserter(result.dimensions),
                   [&](const auto& x) -> dimension { return {{x}}; });
    result.rank = this->rank;
    result.simplify();
    return result;
}
shape_transform_descriptor shape_transform_descriptor::to_common_from_dst() const
{
    shape_transform_descriptor result;
    result.rank = this->dimensions.size();
    std::vector<dimension::sub> subs;
    // Update axes to point to the destination
    for(std::size_t i : range(dimensions.size()))
    {
        const auto& d = dimensions[i];
        const bool mixed_visibility =
            std::any_of(d.subdimensions.begin(),
                        d.subdimensions.end(),
                        [](const dimension::sub& s) { return s.has_hidden_axis(); }) and
            std::any_of(d.subdimensions.begin(),
                        d.subdimensions.end(),
                        [](const dimension::sub& s) { return not s.has_hidden_axis(); });
        std::transform(d.subdimensions.begin(),
                       d.subdimensions.end(),
                       range(d.subdimensions.size()).begin(),
                       std::back_inserter(subs),
                       [&](dimension::sub s, auto j) {
                           set_origin_axis(s, {i});
                           s.add_split_axis(j);
                           if(not mixed_visibility)
                               s.expose();
                           return s;
                       });
    }
    if(dimensions.size() == 1 and subs.empty())
    {
        transform(range(rank), std::back_inserter(subs), [](std::size_t i) -> dimension::sub {
            return {1, {0, i}};
        });
    }
    std::transform(subs.begin(),
                   subs.end(),
                   std::back_inserter(result.dimensions),
                   [&](const auto& x) -> dimension { return {{x}}; });
    renumber_axes(result.dimensions);
    return result;
}
shape_transform_descriptor shape_transform_descriptor::to_dst_from_common() const
{
    shape_transform_descriptor result = *this;
    result.rank                       = result.common_rank();
    if(result.rank == 0)
    {
        result.rank = 1;
        std::vector<dimension::sub> subs;
        transform(range(rank), std::back_inserter(subs), [](std::size_t i) -> dimension::sub {
            return {1, {i}};
        });
        result.dimensions.push_back({subs});
    }
    else
    {
        for_each_subdimension(result.dimensions, range(result.rank), [&](auto& s, std::size_t i) {
            set_origin_axis(s, {i});
            s.expose();
        });
        result.simplify();
    }
    return result;
}
shape_transform_descriptor shape_transform_descriptor::to_src_from_common() const
{
    shape_transform_descriptor result;
    auto subs   = get_all_subdimensions(dimensions);
    result.rank = subs.size();
    transform(group_axes(subs), std::back_inserter(result.dimensions), [&](auto&& p) -> dimension {
        const auto& [axis, gsubs] = p;
        std::vector<dimension::sub> subdimensions;
        transform(gsubs, std::back_inserter(subdimensions), [&](const dimension::sub* s) {
            dimension::sub result = *s;
            std::size_t i         = s - subs.data();
            set_origin_axis(result, {i});
            result.expose();
            return result;
        });
        return {subdimensions};
    });
    result.simplify();
    return result;
}

std::vector<std::vector<std::size_t>> shape_transform_descriptor::common_axes_map_from_src() const
{
    std::vector<std::vector<std::size_t>> result;
    auto subs = get_all_subdimensions(dimensions);
    std::map<std::size_t, std::vector<const dimension::sub*>> axes_map;
    for(const auto& s : subs)
    {
        if(not s.origin_axis().empty())
            axes_map[s.origin_axis().front()].push_back(&s);
    }
    for(auto&& p : axes_map)
    {
        std::sort(p.second.begin(), p.second.end(), by(std::less<>{}, [](const dimension::sub* s) {
                      return s->axis;
                  }));
    }
    if(axes_map.empty() and dimensions.size() == 1)
    {
        transform(range(rank), std::back_inserter(result), [](std::size_t i) {
            return std::vector<std::size_t>{i};
        });
        return result;
    }
    assert(not axes_map.empty());
    auto max_axis = std::prev(axes_map.end())->first;
    result.resize(max_axis + 1);
    for(auto&& p : axes_map)
    {
        assert(p.first < result.size());
        std::transform(p.second.begin(),
                       p.second.end(),
                       std::back_inserter(result[p.first]),
                       [&](const dimension::sub* s) { return s - subs.data(); });
    }
    return result;
}
std::vector<std::vector<std::size_t>> shape_transform_descriptor::common_axes_map_from_dst() const
{
    std::vector<std::vector<std::size_t>> result;
    std::size_t start = 0;
    for(const auto& d : dimensions)
    {
        auto& v = result.emplace_back(d.subdimensions.size());
        std::iota(v.begin(), v.end(), start);
        start += d.subdimensions.size();
    }
    if(result.size() == 1 and result.front().empty())
    {
        result.front().resize(rank);
        std::iota(result.front().begin(), result.front().end(), 0);
    }
    return result;
}

std::vector<std::size_t> shape_transform_descriptor::get_dst_axes_from_src(std::size_t axis) const
{
    std::vector<std::size_t> result;
    for(auto i : range(dimensions.size()))
    {
        const auto& d = dimensions[i];
        auto it       = std::find_if(d.subdimensions.begin(), d.subdimensions.end(), [&](auto& s) {
            if(s.axis.empty())
                return false;
            return s.axis.front() == axis;
        });
        if(it == d.subdimensions.end())
            continue;
        // If it maps to a subdimension then exit as there isn't a clear mapping
        if(d.len() != it->len)
            return {};
        result.push_back(i);
    }
    // TODO: Put it in the correct order if there is multiple axes
    return result;
}

bool shape_transform_descriptor::empty() const { return dimensions.empty(); }

std::vector<std::size_t> shape_transform_descriptor::lens() const
{
    std::vector<std::size_t> result;
    std::transform(dimensions.begin(),
                   dimensions.end(),
                   std::back_inserter(result),
                   [](const dimension& d) { return d.len(); });
    return result;
}

std::size_t dimension::len() const
{
    return transform_accumulate(subdimensions.begin(),
                                subdimensions.end(),
                                std::size_t{1},
                                std::multiplies<>{},
                                [](const auto& s) { return s.len; });
}

std::size_t shape_transform_descriptor::elements() const
{
    return transform_accumulate(dimensions.begin(),
                                dimensions.end(),
                                std::size_t{1},
                                std::multiplies<>{},
                                [](const auto& s) { return s.len(); });
}
std::vector<std::size_t>
shape_transform_descriptor::common_dims(const std::vector<std::size_t>& input_dims) const
{
    std::vector<std::size_t> result;
    for(const auto& d : dimensions)
    {
        std::transform(d.subdimensions.begin(),
                       d.subdimensions.end(),
                       std::back_inserter(result),
                       [&](const dimension::sub& s) { return get_len(s, input_dims); });
    }
    if(result.empty())
        result.resize(rank, 1);
    return result;
}

std::size_t shape_transform_descriptor::common_rank() const
{
    return transform_accumulate(dimensions.begin(),
                                dimensions.end(),
                                std::size_t{0},
                                std::plus<>{},
                                [&](const dimension& d) { return d.subdimensions.size(); });
}

const std::vector<std::size_t>& shape_transform_descriptor::dimension::sub::origin_axis() const
{
    assert(axis.empty() or hidden_axis.empty());
    return axis.empty() ? hidden_axis : axis;
}
bool shape_transform_descriptor::dimension::sub::has_hidden_axis() const
{
    return axis.empty() and not hidden_axis.empty();
}

void shape_transform_descriptor::dimension::sub::add_split_axis(std::size_t i)
{
    assert(axis.empty() or hidden_axis.empty());
    if(not axis.empty())
        axis.push_back(i);
    if(not hidden_axis.empty())
        hidden_axis.push_back(i);
}

void shape_transform_descriptor::dimension::sub::expose()
{
    assert(axis.empty() or hidden_axis.empty());
    if(has_hidden_axis())
    {
        axis = hidden_axis;
        hidden_axis.clear();
    }
}

void shape_transform_descriptor::dimension::sub::hide()
{
    assert(axis.empty() or hidden_axis.empty());
    if(not has_hidden_axis())
    {
        hidden_axis = axis;
        axis.clear();
    }
}

bool operator==(const dimension::sub& x, const dimension::sub& y)
{
    return by(std::equal_to<>{},
              [](const dimension::sub& s) { return std::tie(s.len, s.axis, s.hidden_axis); })(x, y);
}
bool operator!=(const dimension::sub& x, const dimension::sub& y) { return not(x == y); }
std::ostream& operator<<(std::ostream& os, const dimension::sub& x)
{
    os << x.len << ":" << to_string_range(x.axis, "x");
    if(not x.hidden_axis.empty())
        os << "$" << to_string_range(x.hidden_axis, "x");
    return os;
}
bool operator==(const dimension& x, const dimension& y)
{
    return x.subdimensions == y.subdimensions;
}
bool operator!=(const dimension& x, const dimension& y) { return not(x == y); }
std::ostream& operator<<(std::ostream& os, const dimension& x)
{
    os << '[' << stream_range(x.subdimensions) << ']';
    return os;
}
bool operator==(const shape_transform_descriptor& x, const shape_transform_descriptor& y)
{
    return by(std::equal_to<>{}, [](const shape_transform_descriptor& sd) {
        return std::tie(sd.dimensions, sd.rank);
    })(x, y);
}
bool operator!=(const shape_transform_descriptor& x, const shape_transform_descriptor& y)
{
    return not(x == y);
}
std::ostream& operator<<(std::ostream& os, const shape_transform_descriptor& x)
{
    stream_write_value(os, x.dimensions);
    return os;
}

std::vector<operation> optimize_shape_transforms(const std::vector<std::size_t>& dims,
                                                 const std::vector<operation>& ops)
{
    auto sd = shape_transform_descriptor::create(dims, ops);
    if(sd.empty())
        return ops;
    return sd.generate();
}

// Replace broadcasted dimensions with size 1, and set the stride to the previous stride
static shape unbroadcast(const shape& s)
{
    std::vector<std::size_t> lens    = s.lens();
    std::vector<std::size_t> strides = s.strides();
    auto stride_it                   = std::find_if(
        s.strides().begin(), s.strides().end(), [](auto stride) { return stride != 0; });
    std::size_t prev_stride = stride_it == s.strides().end() ? 1 : *stride_it;
    for(std::size_t i = 0; i < lens.size(); ++i)
    {
        if(strides[i] == 0)
        {
            lens[i]    = 1;
            strides[i] = prev_stride;
        }
        else
        {
            prev_stride = strides[i];
        }
    }
    return {s.type(), lens, strides};
}

static std::size_t adjust_strided_shape(shape& s, std::size_t n)
{
    auto lens    = s.lens();
    auto strides = s.strides();

    // Insert a dim of 1 so it can be used to handle steps
    if(std::none_of(strides.begin(), strides.end(), [](auto stride) { return stride == 1; }) and
       std::any_of(strides.begin(), strides.end(), [](auto stride) { return stride != 0; }))
    {
        lens.push_back(1);
        strides.push_back(1);
    }

    auto last_axis      = std::max_element(strides.begin(), strides.end()) - strides.begin();
    auto total_elements = std::max<std::size_t>(1, strides[last_axis] * lens[last_axis]);
    // Add a dim of 1 to the front so it can handle extra elements
    auto extra = n / total_elements;
    if(extra > 1)
    {
        strides.insert(strides.begin(), total_elements);
        lens.insert(lens.begin(), 1);
    }
    s = shape(s.type(), lens, strides);
    return std::max<std::size_t>(1, extra);
}

template <class Range>
static std::vector<std::size_t> select_mask(const std::vector<std::size_t>& slice_mask,
                                            const Range& r)
{
    std::vector<std::size_t> result;
    std::transform(slice_mask.begin(),
                   slice_mask.end(),
                   r.begin(),
                   join_back_inserter(result),
                   [](std::size_t mask, std::size_t n) -> std::vector<std::size_t> {
                       if(mask == 0)
                           return {};
                       return {n};
                   });
    return result;
}

// Generate the shape transforms for strided view
optional<std::vector<operation>>
generate_shape_transforms_for(shape s, const std::vector<std::size_t>& idims, std::int64_t offset)
{
    std::vector<operation> result;
    if(s.lens().empty())
        return std::nullopt;
    std::size_t ielements =
        std::accumulate(idims.begin(), idims.end(), std::size_t(1), std::multiplies<>());
    auto extra = adjust_strided_shape(s, ielements);
    // TODO: Improve handling of multiple dimensions, for now just reshape to 1 dimension
    if(idims.size() != 1)
    {
        result.push_back(make_op("reshape", {{"dims", {ielements}}}));
        auto ops = generate_shape_transforms_for(s, {ielements}, offset);
        if(not ops)
            return std::nullopt;
        result.insert(result.end(), ops->begin(), ops->end());
        return result;
    }
    auto pre_broadcast = unbroadcast(s);
    auto perm          = find_permutation(pre_broadcast);
    auto iperm         = invert_permutation(perm);
    auto pre_transpose = reorder_shape(pre_broadcast, perm);

    std::vector<std::size_t> start_lens;
    std::adjacent_difference(pre_transpose.strides().begin(),
                             pre_transpose.strides().end(),
                             std::back_inserter(start_lens),
                             [](auto y, auto x) -> std::size_t {
                                 assert(x >= y);
                                 assert(y != 0);
                                 if((x % y) != 0)
                                     return 0;
                                 return x / y;
                             });
    if(std::any_of(start_lens.begin(), start_lens.end(), [](auto len) { return len == 0; }))
        return std::nullopt;
    start_lens.front() = extra > 1 ? extra : pre_transpose.lens().front();

    std::size_t nelements =
        std::accumulate(start_lens.begin(), start_lens.end(), std::size_t(1), std::multiplies<>());

    if(nelements < pre_transpose.elements() * extra)
        return std::nullopt;

    std::vector<std::size_t> start_mask(start_lens.size(), 0);
    if(offset != 0)
    {
        shape start_shape{shape::float_type, start_lens};
        auto idx = start_shape.multi(offset);

        std::vector<std::size_t> overhead;
        std::transform(start_lens.begin(),
                       start_lens.end(),
                       pre_transpose.lens().begin(),
                       std::back_inserter(overhead),
                       [](auto start_len, auto len) { return start_len - len; });
        if(std::equal(
               idx.begin(), idx.end(), overhead.begin(), overhead.end(), [](auto i, auto over) {
                   return i <= over;
               }))
        {
            start_mask = reorder_dims(idx, iperm);
            offset     = 0;
        }
    }

    std::vector<std::size_t> pre_slice_mask;
    std::transform(start_lens.begin(),
                   start_lens.end(),
                   pre_transpose.lens().begin(),
                   std::back_inserter(pre_slice_mask),
                   [](auto start_len, auto len) -> std::size_t {
                       if(start_len == len)
                           return 0;
                       return len;
                   });
    auto slice_mask = reorder_dims(pre_slice_mask, iperm);

    std::vector<std::size_t> blens = reorder_dims(start_lens, iperm);
    std::transform(s.lens().begin(),
                   s.lens().end(),
                   blens.begin(),
                   blens.begin(),
                   [](auto len, auto blen) -> std::size_t {
                       if(blen == 1)
                           return len;
                       return blen;
                   });

    std::vector<operation> ops;
    ops.push_back(make_op("multibroadcast", {{"out_lens", blens}}));
    ops.push_back(make_op("transpose", {{"permutation", invert_permutation(perm)}}));
    ops.push_back(make_op("reshape", {{"dims", start_lens}}));
    std::reverse(ops.begin(), ops.end());

    auto desc = shape_transform_descriptor::create({nelements}, ops);

    auto end = offset + nelements;
    if(offset != 0 or nelements != ielements)
    {

        // If the end is out of bounds broadcast it to pad it
        if(end > ielements)
        {
            result.push_back(make_op("broadcast", {{"axis", 1}, {"out_lens", {2, ielements}}}));
            result.push_back(make_op("reshape", {{"dims", {2 * ielements}}}));
        }
        result.push_back(make_op("slice", {{"axes", {0}}, {"starts", {offset}}, {"ends", {end}}}));
    }

    auto opt_ops = desc.generate();
    result.insert(result.end(), opt_ops.begin(), opt_ops.end());

    std::vector<std::size_t> axes = select_mask(slice_mask, range(slice_mask.size()));

    if(not axes.empty())
    {
        std::vector<std::size_t> starts = select_mask(slice_mask, start_mask);
        std::vector<std::size_t> ends   = select_mask(slice_mask, s.lens());
        std::transform(ends.begin(), ends.end(), starts.begin(), ends.begin(), std::plus<>{});

        result.push_back(make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}));
    }
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
