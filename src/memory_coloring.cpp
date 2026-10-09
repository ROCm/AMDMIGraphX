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
#include <migraphx/memory_coloring.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/liveness.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DEBUG_MEMORY_COLORING);

using instruction_set     = std::unordered_set<instruction_ref>;
using instruction_set_map = std::unordered_map<instruction_ref, instruction_set>;

// This will build the conflict table or interference graph. This is
// essentially a map from one instruction to a set of instruction that are
// used together. Each instruction will be the allocation instruction.
static instruction_set_map build_conflict_table(const module& m, const std::string& allocation_op)
{
    instruction_set_map conflict_table;
    liveness(m, [&](auto ins, const auto& live_set) {
        // Skip variables that aren't allocations
        if(ins->name() != allocation_op)
            return;
        // Skip zero allocations
        if(ins->get_shape().bytes() == 0)
            return;
        conflict_table[ins];
        for(auto i : live_set)
        {
            if(i == ins)
                continue;
            // Skip variables that aren't allocations
            if(i->name() != allocation_op)
                continue;
            // Skip zero allocations
            if(i->get_shape().bytes() == 0)
                continue;
            conflict_table[i].insert(ins);
            conflict_table[ins].insert(i);
        }
    });
    assert(std::all_of(conflict_table.begin(), conflict_table.end(), [](auto&& pp) {
        return pp.second.count(pp.first) == 0;
    }));
    return conflict_table;
}

// Check if intervals overlap
static bool is_overlap(std::pair<std::size_t, std::size_t> x, std::pair<std::size_t, std::size_t> y)
{
    return std::max(x.first, y.first) < std::min(x.second, y.second);
}

struct allocation_segment
{
    using segment = std::pair<std::size_t, std::size_t>;
    std::unordered_map<instruction_ref, segment> ins2segment;

    const segment* add_segment(instruction_ref ins, segment s) { return &(ins2segment[ins] = s); }

    const segment* get_segment(instruction_ref ins) const
    {
        auto it = ins2segment.find(ins);
        if(it == ins2segment.end())
            return nullptr;
        return &it->second;
    }

    // Remove segment for an instruction
    void remove(instruction_ref ins)
    {
        auto it = ins2segment.find(ins);
        if(it != ins2segment.end())
        {
            ins2segment.erase(it);
        }
    }

    std::size_t max()
    {
        std::size_t n = 0;
        for(auto&& pp : ins2segment)
        {
            auto seg = pp.second;
            n        = std::max(n, seg.second);
        }
        return n;
    }

    template <class Iterator>
    static bool overlaps(Iterator first, Iterator last, const segment& s)
    {
        return std::any_of(first, last, [&](auto&& t) { return is_overlap(s, t); });
    }

    static bool overlaps(const std::set<segment>& segments, const segment& s)
    {
        return overlaps(segments.begin(), segments.end(), s);
    }

    static auto find_gap(const std::set<segment>& segments, std::size_t n)
    {
        std::size_t max_end = 0;
        return std::adjacent_find(segments.begin(), segments.end(), [&](segment x, segment y) {
            if(x.second < max_end)
                return false;
            max_end = x.second;
            if(is_overlap(x, y))
                return false;
            assert(y.first >= x.second);
            auto k = y.first - x.second;
            return (k >= n);
        });
    }

    static std::size_t max_type_size(const shape& s)
    {
        return std::accumulate(
            s.sub_shapes().begin(),
            s.sub_shapes().end(),
            s.type_size(),
            [](auto size, const auto& sub) { return std::max(size, max_type_size(sub)); });
    }

    static std::size_t compute_alignment(instruction_ref ins)
    {
        auto alignment = max_type_size(ins->get_shape());
        // A rough estimate for the total number of elements
        auto n = ins->get_shape().bytes() / alignment;
        // Check for vectorized alignment
        if(n > 4)
        {
            auto d = n % 4;
            if(d == 0)
                alignment *= 4;
            if(d == 2)
                alignment *= 2;
        }
        return alignment;
    }

    static segment
    next_segment(std::set<segment>& segments, instruction_ref ins, std::size_t alignment)
    {
        assert(ins->get_shape().bytes() > 0);
        // Compute alignment
        std::size_t n = 1 + (ins->get_shape().bytes() - 1) / alignment;
        assert(n > 0);
        std::size_t start = 0;
        // Insert at end if it can't fit at the beginning
        if(segments.empty() or segments.begin()->first <= n)
        {
            auto it = find_gap(segments, n);
            if(it == segments.end())
                it = std::max_element(segments.begin(), segments.end(), [&](segment x, segment y) {
                    return x.second < y.second;
                });
            if(it != segments.end())
                start = it->second;
        }
        auto s = segment{start, start + n};
        assert(not overlaps(segments, s));
        segments.insert(s);
        return s;
    }

    static std::unordered_map<instruction_ref, int>
    create_allocation_index(const module& m, const instruction_set_map& conflict_table)
    {
        std::unordered_map<instruction_ref, int> result;
        int i = 0;
        for(auto ins : iterator_for(m))
        {
            if(not contains(conflict_table, ins))
                continue;
            result[ins] = i++;
        }
        return result;
    }

    // Build the allocation_color class from the conflict_table
    static allocation_segment
    build(const module& m, const instruction_set_map& conflict_table, std::size_t alignment)
    {
        allocation_segment as{};
        std::vector<instruction_ref> conflict_queue;
        // Add all allocations to the conflict_queue
        std::transform(conflict_table.begin(),
                       conflict_table.end(),
                       std::back_inserter(conflict_queue),
                       [](auto&& pp) { return pp.first; });

        auto alloc_index = create_allocation_index(m, conflict_table);

        // Sort the conflict queue so we process the allocation with the most
        // number of adjacent allocations first
        std::sort(conflict_queue.begin(), conflict_queue.end(), by(std::greater<>{}, [&](auto x) {
                      return std::make_tuple(
                          conflict_table.at(x).size(), x->get_shape().bytes(), alloc_index.at(x));
                  }));
        // Process the conflict_queue, we refer to the current allocation as
        // the parent and the adjacent allocations as children
        for(auto parent : conflict_queue)
        {
            // Sort children by size
            std::vector<instruction_ref> children(conflict_table.at(parent).begin(),
                                                  conflict_table.at(parent).end());
            std::sort(children.begin(), children.end(), by(std::less<>{}, [&](auto x) {
                          return std::make_tuple(x->get_shape().bytes(), alloc_index.at(x));
                      }));
            assert(not contains(children, parent));
            // This set is to track the segments already processed
            std::set<segment> segments;
            // Add all segments for the children to the segments already processed
            transform_if(
                children.begin(),
                children.end(),
                std::inserter(segments, segments.begin()),
                [&](auto child) { return as.get_segment(child); },
                [&](auto child) { return *as.get_segment(child); });

            assert(as.get_segment(parent) == nullptr);
            as.add_segment(parent, next_segment(segments, parent, alignment));
        }
        // Reduce the number of segments
        for(std::size_t n = 0; n < 3; n++)
        {
            for(auto parent : conflict_queue)
            {
                auto children = conflict_table.at(parent);
                // This set is to track the segments already processed
                std::set<segment> segments;
                // Add all segments for the children to the segments already processed
                transform_if(
                    children.begin(),
                    children.end(),
                    std::inserter(segments, segments.begin()),
                    [&](auto child) { return as.get_segment(child); },
                    [&](auto child) { return *as.get_segment(child); });
                // Get the segment for the parent
                const auto* parent_segment = as.get_segment(parent);
                assert(parent_segment != nullptr);

                auto s = next_segment(segments, parent, alignment);
                if(s != *parent_segment and s.second <= as.max())
                {
                    as.add_segment(parent, s);
                }
            }
        }
        return as;
    }
};

static std::size_t find_max_alignment(const module& m, const std::string& allocation_op)
{
    std::size_t alignment = 1;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != allocation_op)
            continue;
        alignment = std::max(allocation_segment::compute_alignment(ins), alignment);
    }
    return alignment;
}

template <class Iterator, class F>
static std::size_t max_of(Iterator first, Iterator last, std::size_t init, F f)
{
    return transform_accumulate(
        first, last, init, [](std::size_t x, std::size_t y) { return std::max(x, y); }, f);
}

// The alignment the loads from a scratch parameter were colored with
static std::size_t compute_scratch_alignment(instruction_ref scratch)
{
    const auto& loads = scratch->outputs();
    return max_of(loads.begin(), loads.end(), 1, &allocation_segment::compute_alignment);
}

// Replace the scratch parameters of the submodules of `ins` with a single allocation in the
// parent module, sized for the largest, so they get colored with the parent's allocations. The
// submodules reference the parent allocation directly, which liveness treats as an implicit use
// by `ins`. Returns the alignment the hoisted scratch requires.
static std::size_t
hoist_submodule_scratch(module& m, instruction_ref ins, const allocation_model& model)
{
    // The submodule along with its scratch parameter
    std::vector<std::pair<module_ref, instruction_ref>> params;
    const auto& mods = ins->module_inputs();
    transform_if(
        mods.begin(),
        mods.end(),
        std::back_inserter(params),
        [](module_ref mod) { return mod->get_parameter("scratch") != mod->end(); },
        [](module_ref mod) { return std::make_pair(mod, mod->get_parameter("scratch")); });
    if(params.empty())
        return 1;
    auto bytes     = max_of(params.begin(), params.end(), 0, [](const auto& p) {
        return p.second->get_shape().bytes();
    });
    auto alignment = max_of(params.begin(), params.end(), 1, [](const auto& p) {
        return compute_scratch_alignment(p.second);
    });
    auto alloc     = m.insert_instruction(ins, model.allocate(shape{shape::int8_type, {bytes}}));
    std::for_each(params.begin(), params.end(), [&](const auto& p) {
        p.first->replace_instruction(p.second, alloc);
        p.first->remove_instruction(p.second);
    });
    return alignment;
}

static std::size_t hoist_submodules_scratch(module& m, const allocation_model& model)
{
    std::vector<instruction_ref> parents;
    auto r = iterator_for(m);
    std::copy_if(r.begin(), r.end(), std::back_inserter(parents), [](instruction_ref ins) {
        return not ins->module_inputs().empty();
    });
    return max_of(parents.begin(), parents.end(), 1, [&](instruction_ref ins) {
        return hoist_submodule_scratch(m, ins, model);
    });
}

void memory_coloring::apply(module& m) const
{
    const auto allocation_op        = model.name();
    const std::size_t sub_alignment = hoist_submodules_scratch(m, model);
    const std::size_t alignment     = std::max(find_max_alignment(m, allocation_op), sub_alignment);
    auto conflict_table             = build_conflict_table(m, allocation_op);
    auto as                         = allocation_segment::build(m, conflict_table, alignment);

    // All allocations should have a segment
    assert(std::all_of(conflict_table.begin(), conflict_table.end(), [&](auto&& pp) {
        return as.get_segment(pp.first);
    }));

    // Adjacent allocations should not have overlapping segments
    assert(std::none_of(conflict_table.begin(), conflict_table.end(), [&](auto&& pp) {
        auto* x = as.get_segment(pp.first);
        return std::any_of(pp.second.begin(), pp.second.end(), [&](auto ins) {
            auto* y = as.get_segment(ins);
            assert(x and y);
            return is_overlap(*x, *y);
        });
    }));

    // Print out segments
    if(enabled(MIGRAPHX_DEBUG_MEMORY_COLORING{}))
    {
        for(auto&& pp : conflict_table)
        {
            std::cout << "------- conflict -------" << std::endl;
            auto s1 = as.ins2segment.at(pp.first);
            std::cout << s1.first << ", " << s1.second << ": ";
            m.debug_print(pp.first);
            for(auto ins : pp.second)
            {
                auto s2 = as.ins2segment.at(ins);
                std::cout << s2.first << ", " << s2.second << ": ";
                m.debug_print(ins);
            }
        }
    }

    // Total memory
    std::size_t n = as.max() * alignment;

    // Replace allocations
    auto mem = m.add_parameter("scratch", shape{shape::int8_type, {n}});
    for(auto&& [ins, seg] : as.ins2segment)
    {
        assert(ins->name() == allocation_op);
        auto s             = ins->get_shape();
        std::size_t offset = seg.first * alignment;
        assert(offset < n);
        m.replace_instruction(
            ins, make_op("load", {{"shape", to_value(s)}, {"offset", offset}}), mem);
    }

    // Replace zero allocation
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != allocation_op)
            continue;
        assert(ins->get_shape().bytes() == 0);
        m.replace_instruction(
            ins, make_op("load", {{"shape", to_value(ins->get_shape())}, {"offset", 0}}), mem);
    }

    // Remove scratch parameter if its not used
    if(mem->outputs().empty())
    {
        m.remove_instruction(mem);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
