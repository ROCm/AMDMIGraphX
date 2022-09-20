/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>

#include "memory_coloring_impl.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void memory_coloring_impl::run()
{
    // calc implicit depdendencies
    mod_implicit_deps = p_mod->calc_implicit_deps();

    MIGRAPHX_DEBUG(dump("---Before memory coloring---"));
    MIGRAPHX_DEBUG(dump_module());
    build();
    if(num_of_lives != 0)
    {
        MIGRAPHX_DEBUG(dump_intervals());
        // Coloring
        while(not alloc_queue.empty())
        {
            interval_ptr interval = alloc_queue.top();
            allocate(interval);
            alloc_queue.pop();
        }

        // rewrite happens after all modules are processed
        rewrite();

        if(enable_verify)
            verify();
    }
}

bool memory_coloring_impl::allocate(interval_ptr interval)
{
    shape s          = interval->result;
    std::size_t size = s.bytes();
    if(size == 0)
        return false;
    std::size_t element_size = (s.elements() == 0 ? 4 : (size / s.elements()));
    live_range& segment      = interval->segment;
    int vn                   = segment.vn;
    std::priority_queue<live_range*, std::vector<live_range*>, ordering> conflict_queue;
    std::unordered_map<long long, live_range*> offset2_live;
    offset2_live.clear();

    if(conflict_table.find(vn) != conflict_table.end())
    {
        std::set<int>& vn_set = conflict_table[vn];
        for(const auto& iter : vn_set)
        {
            live_range* range = live_ranges[iter];
            long long offset  = range->offset;
            if(offset != invalid_offset)
            {
                conflict_queue.push(range);
                if(offset2_live.find(offset) == offset2_live.end())
                {
                    offset2_live[offset] = range;
                }
                else
                {
                    live_range* prev = offset2_live[offset];
                    assert(prev->offset == offset);
                    if(prev->size < range->size)
                        offset2_live[offset] = range;
                }
            }
        }
    }

    std::size_t offset = 0;
    while(not conflict_queue.empty())
    {
        live_range* range       = conflict_queue.top();
        std::size_t iter_offset = range->offset;
        if(offset > iter_offset)
        {
            offset = std::max(offset, iter_offset + range->size);
        }
        else if(offset2_live[iter_offset] == range)
        {
            if((iter_offset > offset) && (iter_offset - offset) >= size)
            {
                break;
            }
            offset = iter_offset + range->size;
        }
        // alignment
        if((offset % element_size) != 0)
            offset += (element_size - (offset % element_size));
        conflict_queue.pop();
    }
    // when int8 type is used, the offset could be any number
    // if not 4-byte aligned, miopen int8 convolution can crash
    offset         = (offset + 3) / 4 * 4;
    segment.offset = offset;
    MIGRAPHX_DEBUG(segment.dump());
    required_bytes = std::max(required_bytes, offset + segment.size);
    return true;
}

void memory_coloring_impl::build()
{
    std::size_t num_of_instrs = p_mod->size();
    if(num_of_instrs == 0)
        return;

    auto cur_points       = num_of_instrs * 2;
    instruction_ref iter  = p_mod->end();
    instruction_ref begin = p_mod->begin();
    std::vector<instruction_ref> dead_instrs;
    std::set<int> live_set;
    // Build live intervals.
    live_intervals.resize(num_of_instrs);
    do
    {
        iter                      = std::prev(iter);
        const instruction* p_iter = &(*iter);
        interval_ptr def_interval = nullptr;
        bool is_dead              = false;
        if(instr2_live.find(p_iter) != instr2_live.end())
        {
            def_interval = instr2_live[p_iter];
            bool is_lit  = is_literal(iter);
            if(is_allocate(iter) or is_lit)
            {
                live_range& range        = def_interval->segment;
                def_interval->result     = iter->get_shape();
                def_interval->is_literal = is_lit;
                range.begin              = cur_points;
                def_interval->def_point  = cur_points;
                range.size               = (iter->get_shape()).bytes();
                if(not is_lit or unify_literals)
                    alloc_queue.push(def_interval);
                live_set.erase(range.vn);
            }
        }
        else if(not is_param(iter) && not is_outline(iter) && not is_check_context(iter))
        {
            is_dead = true;
        }

        auto inputs = iter->inputs();
        if(contains(mod_implicit_deps, iter))
        {
            const auto& impl_deps = mod_implicit_deps.at(iter);
            inputs.insert(inputs.end(), impl_deps.begin(), impl_deps.end());
        }

        for(auto&& arg : inputs)
        {
            if(not p_mod->has_instruction(arg))
                continue;

            if(is_param(arg) or is_outline(arg))
            {
                if(is_output_param(arg))
                    is_dead = false;
                if(def_interval != nullptr)
                {
                    def_interval->is_live_on_entry = true;
                }
                continue;
            }
            const instruction* p_arg = &(*instruction::get_output_alias(arg));
            if(instr2_live.find(p_arg) == instr2_live.end())
            {
                // First time see a use, create a live interval.
                int id                = num_of_lives++;
                interval_ptr interval = &(live_intervals[id]);
                interval->id          = id;
                interval->segment.end = cur_points;
                interval->segment.vn  = ++max_value_number;
                interval->add_use(cur_points);
                instr2_live[p_arg] = interval;
                add_conflicts(live_set, max_value_number);
                live_set.insert(max_value_number);
                live_ranges[max_value_number] = &(interval->segment);
                earliest_end_point            = cur_points;
                if(latest_end_point == -1)
                    latest_end_point = cur_points;
            }
            else
            {
                interval_ptr interval = instr2_live[p_arg];
                interval->add_use(cur_points);
                assert(live_set.find(interval->id) != live_set.end());
            }
        }
        if(is_dead)
            dead_instrs.push_back(iter);
        cur_points -= 2;
    } while(iter != begin);
}

void memory_coloring_impl::rewrite()
{
    std::vector<std::size_t> dims;
    dims.push_back((required_bytes + sizeof(float) - 1) / sizeof(float));
    shape s                       = {shape::float_type, dims};
    instruction_ref scratch_param = p_mod->add_parameter("scratch", s);
    for(auto ins : iterator_for(*p_mod))
    {
        const instruction* p_iter = &(*ins);
        if(instr2_live.find(p_iter) != instr2_live.end())
        {
            interval_ptr interval = instr2_live[p_iter];
            if(interval->get_begin() == invalid_offset)
                continue;

            if(not unify_literals && interval->is_literal)
                continue;

            std::size_t offset = 0;
            if(interval->get_offset() != invalid_offset)
            {
                offset = interval->get_offset();
            }
            else
            {
                assert(interval->result.bytes() == 0);
            }

            if(is_allocate(ins))
            {
                p_mod->replace_instruction(
                    ins,
                    make_op("load", {{"shape", to_value(ins->get_shape())}, {"offset", offset}}),
                    scratch_param);
            }
        }
    }
    MIGRAPHX_DEBUG(dump("---After rewrite---"));
    MIGRAPHX_DEBUG(dump_module());
}

void memory_coloring_impl::verify()
{
    if(num_of_lives > 0)
    {
        for(int i = 0; i < num_of_lives; ++i)
        {
            live_interval& interval = live_intervals[i];
            live_range& segment     = interval.segment;

            if(segment.begin == invalid_offset)
            {
                // if(not interval.is_live_on_entry)
                // MIGRAPHX_THROW("interval is not live on entry");
                continue;
            }

            if(segment.offset == invalid_offset)
            {
                continue;
            }
            int vn = segment.vn;
            if(conflict_table.find(vn) != conflict_table.end())
            {
                std::set<int>& vn_set = conflict_table[vn];
                for(const auto& iter : vn_set)
                {
                    live_range* range = live_ranges[iter];
                    if(range->offset == invalid_offset)
                        continue;
                    if(not is_disjoin(*range, segment))
                        MIGRAPHX_THROW("range and segment is not disjoined");
                }
            }
        }
    }
}

#ifdef MIGRAPHX_DEBUG_OPT

void memory_coloring_impl::dump(const std::string& str) { std::cout << str << std::endl; }

void memory_coloring_impl::dump_module() { std::cout << *p_mod << std::endl; }

void memory_coloring_impl::dump_intervals()
{
    if(num_of_lives > 0)
    {
        std::cout << "---live intervals ---" << std::endl;
        for(int i = 0; i < num_of_lives; ++i)
        {
            live_interval& interval = live_intervals[i];
            interval.dump();
        }
        std::cout << "---conflict table---" << std::endl;
        for(int i = 0; i <= max_value_number; ++i)
        {
            std::cout << " segment:" << i;
            std::cout << " =>";
            std::set<int>& table = conflict_table[i];
            for(auto& iter : table)
            {
                std::cout << (iter) << ",";
            }
        }
        std::cout << std::endl;
    }
}

// map liveness tracking point to instruction enum.
static int get_ins_enum(int x)
{
    if(x > 0)
    {
        return (x / 2) - 1;
    }
    else
        return invalid_offset;
}

void live_range::dump()
{
    std::cout << " segment:" << vn;
    std::cout << " [" << get_ins_enum(begin) << ", " << get_ins_enum(end) << "]";
    if(offset != invalid_offset)
    {
        std::cout << " mem:";
        std::cout << " [" << offset << "," << offset + size - 1 << "]";
    }
    std::cout << std::endl;
}

void live_interval::dump()
{
    std::cout << "id:" << id;
    segment.dump();
    std::cout << " uses:";
    for(auto& iter : use_points)
    {
        std::cout << " " << get_ins_enum(iter) << ",";
    }
    std::cout << " def:";
    std::cout << " " << get_ins_enum(def_point);

    if(is_literal)
        std::cout << " literal";
    std::cout << " " << result;
    std::cout << std::endl;
}

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
