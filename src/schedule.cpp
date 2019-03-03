#include <migraphx/schedule.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_map>
#include <unordered_set>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool stream_free(instruction_ref ins)
{
    return is_context_free(ins->get_operator()) or ins->get_operator().name().front() == '@';
}

struct stream_info
{
    std::unordered_map<instruction_ref, std::size_t> ins2stream;
    std::unordered_map<instruction_ref, std::size_t> weights;

    void accumulate_weights(instruction_ref last, const schedule_model& model)
    {
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
    }

    void assign_streams(program& p, std::size_t streams)
    {
        const std::size_t min_partition_threshold = 2;
        for(std::size_t stream = 0; stream < streams; stream++)
        {
            fix([&](auto self, auto ins) {
                // If weight is zero then stop
                if(weights[ins] == 0)
                    return;
                // Only assign streams if not already assigned
                if(not has_stream(ins))
                    set_stream(ins, stream);
                instruction_ref child = p.end();
                std::size_t w         = 0;
                for(auto i : ins->inputs())
                {
                    const auto weight = weights[i];
                    // Skip instruction that already have stream assignment or too low of weights
                    if(has_stream(i) or weight <= min_partition_threshold)
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
            })(std::prev(p.end()));
        }
        // Assign remaining instructions
        for(auto ins : iterator_for(p))
        {
            if(has_stream(ins))
                continue;
            if(weights[ins] == 0)
                continue;
            set_stream(ins, streams - 1);
        }
    }

    void set_stream(instruction_ref ins, std::size_t n) { ins2stream[ins] = n; }

    std::size_t get_stream(instruction_ref ins) const { return ins2stream.at(ins); }

    bool has_stream(instruction_ref ins) const { return ins2stream.count(ins) > 0; }

    bool different(const std::vector<std::size_t>& v) const
    {
        if(v.size() < 2)
            return false;
        return not std::all_of(v.begin(), v.end(), [&](std::size_t x) { return x == v.front(); });
    }

    template<class Selector>
    std::vector<std::size_t> get_streams(instruction_ref ins, Selector select) const
    {
        std::vector<std::size_t> result;
        for(auto i : select(ins))
        {
            if(weights.at(i) == 0)
            {
                auto vv = get_input_streams(i);
                result.insert(result.end(), vv.begin(), vv.end());
            }
            else
            {
                result.emplace_back(get_stream(i));
            }
        }
        return result;
    }

    std::vector<std::size_t> get_input_streams(instruction_ref ins) const
    {
        return get_streams(ins, [](auto i) {
            return i->inputs();
        });
    }

    std::vector<std::size_t> get_output_streams(instruction_ref ins) const
    {
        return get_streams(ins, [](auto i) {
            return i->outputs();
        });
    }

    bool is_merge_point(instruction_ref ins) const { return different(get_input_streams(ins)); }
    
    bool is_split_point(instruction_ref ins) const { return different(get_output_streams(ins)); }

    std::vector<std::size_t> wait_for(instruction_ref ins) const
    {
        std::vector<std::size_t> result = get_input_streams(ins);
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

    template<class F>
    void find_concurrent_instructions(program& p, F f)
    {
        std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>> split_from;
        for(auto ins : iterator_for(p))
        {
            if (weights[ins] == 0)
                continue;
            for(auto&& arg : ins->inputs())
            {
                if (is_split_point(arg))
                    split_from[ins].insert(arg);
                split_from[ins].insert(split_from[arg].begin(), split_from[arg].end());
            }

            // Collect concur instructions for each split point.
            for(auto& split : split_from[ins])
            {
                f(ins, split);
            }
        }
    }
};

void schedule::apply(program& p) const
{
    stream_info si;
    auto last = std::prev(p.end());
    si.accumulate_weights(last, model);
    si.assign_streams(p, model.concurrency());

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
        // Only schedule instructions that have a stream
        if(not si.has_stream(ins))
            continue;
        if(si.is_merge_point(ins))
            model.wait(p, ins, si.get_stream(ins), si.wait_for(ins));
        else
            model.schedule_instruction(p, ins, si.get_stream(ins));
    }

    si.find_concurrent_instructions(p, [&](auto x, auto y) {
        p.insert_instruction(std::next(x), op::identity{}, x, y);
    });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
