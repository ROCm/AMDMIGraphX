#include <migraphx/schedule.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_map>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct stream_info
{
    std::unordered_map<instruction_ref, std::size_t> ins2stream;

    void set_stream(instruction_ref ins, std::size_t n) { ins2stream[ins] = n; }

    std::size_t get_stream(instruction_ref ins) const { return ins2stream.at(ins); }

    bool has_stream(instruction_ref ins) const { return ins2stream.count(ins) > 0; }

    bool different(const std::vector<instruction_ref>& v) const
    {
        if(v.size() < 2)
            return false;
        auto stream = get_stream(v.front());
        return not std::all_of(
            v.begin(), v.end(), [&](instruction_ref x) { return get_stream(x) == stream; });
    }

    bool is_split_point(instruction_ref ins) const { return different(ins->outputs()); }

    bool is_merge_point(instruction_ref ins) const { return different(ins->inputs()); }

    std::vector<std::size_t> wait_for(instruction_ref ins) const
    {
        std::set<std::size_t> result;
        auto s = get_stream(ins);
        for(auto i : ins->inputs())
        {
            auto stream = get_stream(i);
            if(stream != s)
                result.insert(stream);
        }
        return {result.begin(), result.end()};
    }
};

void schedule::apply(program& p) const
{
    const std::size_t min_partition_threshold = 2;

    // Compute accumulated weights
    std::unordered_map<instruction_ref, std::size_t> weights;
    auto last = std::prev(p.end());
    fix<std::size_t>([&](auto self, auto ins) -> std::size_t {
        if(weights.count(ins) == 0)
        {
            weights[ins] =
                std::accumulate(ins->inputs().begin(),
                                ins->inputs().end(),
                                model.weight(ins->get_operator()),
                                [&](std::size_t w, instruction_ref i) { return w + self(i); });
        }
        return weights[ins];
    })(last);

    // Assign streams
    auto streams = model.concurrency();
    stream_info si;
    for(std::size_t stream = 0; stream < streams; stream++)
    {
        fix([&](auto self, auto ins) {
            // Only assign streams fi not already assigned
            if(not si.has_stream(ins))
                si.set_stream(ins, stream);
            instruction_ref child = p.end();
            std::size_t w         = 0;
            for(auto i : ins->inputs())
            {
                const auto weight = weights[i];
                // Skip instruction that already have stream assignment or too low of weights
                if(si.has_stream(i) or weight <= min_partition_threshold)
                {
                    self(i);
                }
                // Accumulate the max weight
                else if(weight > w)
                {
                    child = i;
                    w     = weight;
                }
            }
            if(child != p.end())
                self(child);
        })(last);
    }
    // Assign remaining instructions
    for(auto ins : iterator_for(p))
    {
        if(si.has_stream(ins))
            continue;
        si.set_stream(ins, streams - 1);
    }

    // Topo sort
    fix([&](auto self, auto ins) {
        for(auto i : ins->inputs())
            p.move_instruction(i, p.begin());
        for(auto i : ins->inputs())
            self(i);
    })(last);

    // Schedule instructions
    for(auto ins : iterator_for(p))
    {
        if(si.is_merge_point(ins))
        {
            assert(not si.wait_for(ins).empty());
            model.wait(p, ins, si.get_stream(ins), si.wait_for(ins));
            continue;
        }
        // Skip scheduling instructions with no context
        if(is_context_free(ins->get_operator()) or ins->get_operator().name().front() == '@')
            continue;
        model.schedule_instruction(p, ins, si.get_stream(ins));
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
