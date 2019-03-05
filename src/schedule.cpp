#include <migraphx/schedule.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/dfor.hpp>
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

auto get_inputs()
{
    return [](auto i) { return i->inputs(); };
}

auto get_outputs()
{
    return [](auto i) { return i->outputs(); };
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
                std::size_t weight = 0;
                auto&& op          = ins->get_operator();
                if(not is_context_free(op) and op.name()[0] != '@')
                    weight = model.weight(op);
                weights[ins] =
                    std::accumulate(ins->inputs().begin(),
                                    ins->inputs().end(),
                                    weight,
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
                if(this->weights[ins] == 0)
                    return;
                // Only assign streams if not already assigned
                if(not this->has_stream(ins))
                    this->set_stream(ins, stream);
                instruction_ref child = p.end();
                std::size_t w         = 0;
                for(auto i : ins->inputs())
                {
                    const auto weight = this->weights[i];
                    // Skip instruction that already have stream assignment or too low of weights
                    if(this->has_stream(i) or weight <= min_partition_threshold)
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

    template <class F>
    bool different(F f, std::size_t stream) const
    {
        bool result = false;
        f([&](auto s) {
            if(s != stream)
            {
                result = true;
                return false;
            }
            stream = s;
            return true;
        });
        return result;
    }

    template <class F>
    bool different(F f) const
    {
        bool result = false;
        f([&](auto s) {
            result = different(f, s);
            return false;
        });
        return result;
    }

    template <class Selector>
    auto get_streams(instruction_ref start, Selector select) const
    {
        return [=](auto f) {
            return fix<bool>([&](auto self, auto ins) {
                for(auto i : select(ins))
                {
                    if(weights.at(i) == 0)
                    {
                        if(not self(i))
                            return false;
                    }
                    else
                    {
                        if(not f(get_stream(i)))
                            return false;
                    }
                }
                return true;
            })(start);
        };
    }

    template<class... Ts>
    bool is_merge_point(instruction_ref ins, Ts... xs) const { return different(get_streams(ins, get_inputs()), xs...); }

    template<class... Ts>
    bool is_split_point(instruction_ref ins, Ts... xs) const { return different(get_streams(ins, get_outputs()), xs...); }

    std::vector<std::size_t> wait_for(instruction_ref ins) const
    {
        std::vector<std::size_t> result;
        get_streams(ins, get_inputs())([&](auto s) {
            result.push_back(s);
            return true;
        });
        // Remove duplicates
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        // Remove the merged stream
        auto it = std::find(result.begin(), result.end(), get_stream(ins));
        if (it != result.end())
            result.erase(it);
        return result;
    }

    std::unordered_map<instruction_ref, std::vector<std::vector<instruction_ref>>>
    find_concurrent_instructions(program& p)
    {
        std::unordered_map<instruction_ref, std::vector<std::vector<instruction_ref>>> result;
        std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>> split_from;
        for(auto ins : iterator_for(p))
        {
            if(weights[ins] == 0)
                continue;
            for(auto&& arg : ins->inputs())
            {
                if(is_split_point(arg))
                    split_from[ins].insert(arg);
                split_from[ins].insert(split_from[arg].begin(), split_from[arg].end());
            }

            auto stream = get_stream(ins);
            // if (is_merge_point(ins))
            // {
            //     // post-dominator kills split point.
            //     for(auto& split : split_from[ins])
            //     {
            //         if(strictly_post_dominates(ins, split))
            //             split_from[ins].erase(split);
            //     }
            // }

            // Collect concur instructions for each split point.
            for(auto& split : split_from[ins])
            {
                if(result[split].size() <= stream)
                    result[split].resize(stream + 1);
                result[split][stream].push_back(ins);
            }
        }
        return result;
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
        auto args = ins->inputs();
        std::sort(args.begin(), args.end(), [&](auto x, auto y) {
            return si.weights[x] < si.weights[y];
        });
        for(auto i : args)
            p.move_instruction(i, p.begin());
        for(auto i : args)
            self(i);
    })(last);

    if(enabled(MIGRAPHX_TRACE_COMPILE{}))
    {
        p.annotate(std::cout, [&](auto ins) {
            std::cout << ":";
            std::cout << " weight=" << si.weights.at(ins);
            if(si.has_stream(ins))
                std::cout << " stream=" << si.get_stream(ins);
        });
        std::cout << std::endl;
    }

    // Schedule instructions
    for(auto ins : iterator_for(p))
    {
        // Only schedule instructions that have a stream
        if(not si.has_stream(ins))
            continue;
        auto stream = si.get_stream(ins);
        if(si.is_merge_point(ins, stream))
            model.wait(p, ins, stream, si.wait_for(ins));
        else
            model.schedule_instruction(p, ins, stream);
    }

    // Add memory conflicts
    auto concur_ins = si.find_concurrent_instructions(p);
    for(auto&& split : concur_ins)
    {
        dfor(split.second.size(), split.second.size())([&](auto i, auto j) {
            if(i == j)
                return;
            for(auto ins1 : split.second[i])
            {
                auto args = split.second[j];
                args.insert(args.begin(), ins1);

                auto point = std::max_element(args.begin(), args.end(), [&](auto x, auto y) {
                    return std::distance(split.first, x) < std::distance(split.first, y);
                });
                p.insert_instruction(std::next(*point), op::identity{}, args);
            }
        });
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
