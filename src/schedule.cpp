#include <migraphx/schedule.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <thread>
#include <mutex>
#include <set>
#include <deque>
#include <chrono>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_SCHEDULE)

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
    std::unordered_map<instruction_ref, std::size_t> iweights;

    void accumulate_weights(instruction_ref last, const schedule_model& model)
    {
        fix<std::size_t>([&](auto self, auto ins) -> std::size_t {
            if(not contains(weights, ins))
            {
                std::size_t weight = 0;
                auto&& op          = ins->get_operator();
                if(not is_context_free(op) and op.name()[0] != '@')
                    weight = model.weight(op);
                iweights[ins] = weight;
                weights[ins] =
                    std::accumulate(ins->inputs().begin(),
                                    ins->inputs().end(),
                                    weight,
                                    [&](std::size_t w, instruction_ref i) { return w + self(i); });
            }
            return weights[ins];
        })(last);
    }

    template <class Compare>
    auto compare_by_weights(Compare compare) const
    {
        return by(compare, [this](auto x) {
            return std::make_tuple(this->weights.at(x), x->inputs().size(), std::addressof(*x));
        });
    }

    template <class Compare>
    void sort_args_by_weight(std::vector<instruction_ref>& args, Compare compare) const
    {
        if(args.size() < 2)
            return;
        std::sort(args.begin(), args.end(), compare_by_weights(compare));
    }

    std::vector<instruction_ref>::iterator sort_args(std::vector<instruction_ref>& args)
    {
        if(args.size() < 2)
        {
            return args.end();
        }

        const std::size_t min_partition_threshold = 1;
        sort_args_by_weight(args, std::greater<>{});

        auto it = std::lower_bound(std::next(args.begin()),
                                   args.end(),
                                   min_partition_threshold,
                                   [&](auto i, std::size_t w) { return this->weights[i] > w; });
        assert(it == args.end() or this->weights[*it] <= min_partition_threshold);
        assert(it == args.end() or std::prev(it) == args.begin() or
               this->weights[*std::prev(it)] > min_partition_threshold);
        return it;
    }

    struct partition
    {
        std::size_t weight = 0;
        std::vector<instruction_ref> instructions{};

        void add(instruction_ref ins, std::size_t w)
        {
            weight += w;
            instructions.push_back(ins);
        }
    };

    std::size_t assign_streams(program& p, std::size_t n)
    {
        assert(n > 0);
        partition critical;
        std::unordered_map<instruction_ref, std::deque<partition>> partitions;
        partitions.reserve(weights.size());
        fix([&](auto self, auto ins, auto& part) {
            assert(ins != p.end());
            if(contains(partitions, ins))
                return;
            assert(p.has_instruction(ins));
            // Add an entry so we know the instruction was visited
            partitions[ins];
            part.add(ins, this->iweights[ins]);

            auto args         = ins->inputs();
            auto threshold_it = this->sort_args(args);

            if(not args.empty())
            {
                assert(threshold_it != args.begin());
                self(args.front(), part);
                for(auto i : range(std::next(args.begin()), threshold_it))
                {
                    partitions[ins].emplace_back();
                    self(i, partitions[ins].back());
                }
                for(auto i : range(threshold_it, args.end()))
                {
                    self(i, part);
                }
            }
            // Sort instructions
            p.move_instruction(ins, p.end());
        })(std::prev(p.end()), critical);

        // Set the critical partition to stream 0
        set_stream(critical, 0);
        if(n == 1)
        {
            // Assign streams for the other partitions
            for(auto&& ins_part : partitions)
                for(auto&& part : ins_part.second)
                    set_stream(part, 0);
            return 1;
        }
        else
        {
            std::vector<std::size_t> streams(n - 1);
            // Assign streams for the other partitions
            for(auto&& ins_part : partitions)
            {
                std::sort(ins_part.second.begin(),
                          ins_part.second.end(),
                          by(std::greater<>{}, [](auto&& x) {
                              return std::make_tuple(x.weight, x.instructions.size());
                          }));
                for(auto&& part : ins_part.second)
                {
                    auto stream =
                        std::min_element(streams.begin(), streams.end()) - streams.begin();
                    set_stream(part, stream + 1);
                    streams[stream] += part.weight;
                }
            }
            return 1 + std::count_if(streams.begin(), streams.end(), [](auto x) { return x > 0; });
        }
    }

    using weight_ins = std::pair<std::size_t, instruction_ref>;
    struct compare_weight_ins
    {
        bool operator()(const weight_ins& x, const weight_ins& y) const
        {
            return std::make_pair(x.first, std::addressof(*x.second)) >
                   std::make_pair(y.first, std::addressof(*y.second));
        }
    };

    void sort(program& p) const
    {
        std::priority_queue<weight_ins, std::vector<weight_ins>, compare_weight_ins> children;
        auto add_child = [&](auto ins) {
            children.push(std::make_pair(this->iweights.at(ins), ins));
        };
        children.push(std::make_pair(0, std::prev(p.end())));

        std::unordered_set<instruction_ref> visited;

        while(not children.empty())
        {
            // Pop the first element
            auto top = children.top().second;
            children.pop();

            p.move_instruction(top, p.begin());
            for(auto ins : top->inputs())
            {
                add_child(ins);
            }
        }
    }

    void set_stream(const partition& p, std::size_t n)
    {
        for(auto ins : p.instructions)
            if(iweights[ins] > 0)
                set_stream(ins, n);
    }

    void set_stream(instruction_ref ins, std::size_t n)
    {
        assert(iweights[ins] > 0);
        ins2stream[ins] = n;
    }

    std::size_t get_stream(instruction_ref ins) const { return ins2stream.at(ins); }

    bool has_stream(instruction_ref ins) const { return contains(ins2stream, ins); }

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
            // cppcheck-suppress uselessAssignmentArg
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
            result = this->different(f, s);
            return false;
        });
        return result;
    }

    template <class Selector>
    auto get_streams_from(instruction_ref start, Selector select) const
    {
        return [=](auto f) {
            return fix<bool>([&](auto self, auto ins) {
                for(auto i : select(ins))
                {
                    if(iweights.at(i) == 0)
                    {
                        if(not self(i))
                            return false;
                    }
                    else
                    {
                        if(not f(this->get_stream(i)))
                            return false;
                    }
                }
                return true;
            })(start);
        };
    }

    std::unordered_set<std::size_t> get_streams(instruction_ref ins) const
    {
        if(has_stream(ins))
            return {get_stream(ins)};
        std::unordered_set<std::size_t> result;
        get_streams_from(ins, get_inputs())([&](auto s) {
            result.insert(s);
            return true;
        });
        return result;
    }

    template <class... Ts>
    bool is_merge_point(instruction_ref ins, Ts... xs) const
    {
        return different(get_streams_from(ins, get_inputs()), xs...);
    }

    template <class... Ts>
    bool is_split_point(instruction_ref ins, Ts... xs) const
    {
        return different(get_streams_from(ins, get_outputs()), xs...);
    }

    std::vector<instruction_ref> get_recorded_instructions(instruction_ref start)
    {
        std::vector<instruction_ref> result;
        std::unordered_map<std::size_t, instruction_ref> m;
        fix([&](auto self, auto ins) {
            for(auto i : ins->inputs())
            {
                if(iweights.at(i) == 0)
                {
                    self(i);
                    continue;
                }
                auto stream = this->get_stream(i);
                if(not contains(m, stream))
                    m[stream] = i;
                else
                    m[stream] = std::min(m[stream], i, by(std::less<>{}, [&](auto x) {
                                             return std::distance(x, start);
                                         }));
            }
        })(start);
        std::transform(
            m.begin(), m.end(), std::back_inserter(result), [](auto&& p) { return p.second; });
        return result;
    }

    std::unordered_map<instruction_ref, std::vector<std::vector<instruction_ref>>>
    find_concurrent_instructions(program& p)
    {
        std::unordered_map<instruction_ref, std::vector<std::vector<instruction_ref>>> result;
        std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>> merge_from;
        result.reserve(p.size());
        merge_from.reserve(p.size());
        for(auto ins : reverse_iterator_for(p))
        {
            for(auto&& arg : ins->outputs())
            {
                if(is_merge_point(arg))
                    merge_from[ins].insert(arg);
                merge_from[ins].insert(merge_from[arg].begin(), merge_from[arg].end());
            }

            auto streams = this->get_streams(ins);

            // Collect concur instructions for each merge point.
            for(auto& merge : merge_from[ins])
            {
                for(auto stream : streams)
                {
                    if(result[merge].size() <= stream)
                        result[merge].resize(stream + 1);
                    auto&& r = result[merge][stream];
                    r.push_back(ins);
                    // Copy inputs if they dont have a stream(and are not a builtin and context
                    // free). Inputs without a stream can have a implicit dependency
                    std::copy_if(ins->inputs().begin(),
                                 ins->inputs().end(),
                                 std::back_inserter(r),
                                 [&](auto x) {
                                     return not this->has_stream(x) and
                                            not is_context_free(x->get_operator()) and
                                            x->name().front() != '@';
                                 });
                }
            }
        }
        return result;
    }

    std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>>
    get_conflicts(program& p)
    {
        using conflict_table_type =
            std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>>;
        conflict_table_type conflict_table;
        auto concur_ins = this->find_concurrent_instructions(p);

        std::vector<conflict_table_type> thread_conflict_tables(
            std::thread::hardware_concurrency());
        std::vector<instruction_ref> index_to_ins;
        index_to_ins.reserve(concur_ins.size());
        std::transform(concur_ins.begin(),
                       concur_ins.end(),
                       std::back_inserter(index_to_ins),
                       [](auto&& it) { return it.first; });

        par_for(concur_ins.size(), [&](auto ins_index, auto tid) {
            auto merge_first = index_to_ins[ins_index];
            assert(concur_ins.count(merge_first) > 0);
            auto& merge_second = concur_ins.at(merge_first);

            // ensure there are enough elements for different threads
            assert(tid < thread_conflict_tables.size());
            auto& thrd_table = thread_conflict_tables.at(tid);

            std::unordered_set<instruction_ref> checked_ins_set;
            auto range_i = range(merge_second.begin(), std::prev(merge_second.end()));
            for(auto it_i : iterator_for(range_i))
            {
                std::unordered_set<instruction_ref> ins1_set;
                std::copy_if(it_i->begin(),
                             it_i->end(),
                             std::inserter(ins1_set, ins1_set.end()),
                             [&](auto i) { return not contains(checked_ins_set, i); });
                checked_ins_set.insert(ins1_set.begin(), ins1_set.end());

                auto range_j = range(std::next(it_i), merge_second.end());
                std::unordered_set<instruction_ref> ins2_set;
                for(auto it_j : iterator_for(range_j))
                {
                    std::copy_if(it_j->begin(),
                                 it_j->end(),
                                 std::inserter(ins2_set, ins2_set.end()),
                                 [&](auto i) { return not contains(checked_ins_set, i); });
                }

                for(auto ins1 : ins1_set)
                {
                    auto p1 = std::distance(ins1, merge_first);
                    for(auto ins2 : ins2_set)
                    {
                        if(ins1 == ins2)
                            continue;
                        auto p2 = std::distance(ins2, merge_first);
                        // The smaller distance means the instruction occurs later
                        if(p1 > p2)
                            thrd_table[ins2].insert(ins1);
                        else
                            thrd_table[ins1].insert(ins2);
                    }
                }
            }
        });

        // merge thread_conflict_tables together
        for(auto& tbl : thread_conflict_tables)
        {
            for(auto& it : tbl)
            {
                conflict_table[it.first].insert(it.second.begin(), it.second.end());
            }
        }

        // Remove instructions from the conflict table of an ealier instruction
        for(auto&& ip : conflict_table)
        {
            auto ins1 = ip.first;
            for(auto ins2 : ip.second)
                if(contains(conflict_table[ins2], ins1))
                    conflict_table[ins2].erase(ins1);
        }

        return conflict_table;
    }
};

void schedule::apply(program& p) const
{
    if(not enable)
        return;
    stream_info si;
    auto last = std::prev(p.end());
    si.accumulate_weights(last, model);
    auto nstreams = si.assign_streams(p, model.concurrency());
    si.sort(p);

    if(enabled(MIGRAPHX_TRACE_COMPILE{}) or enabled(MIGRAPHX_TRACE_SCHEDULE{}))
    {
        p.annotate(std::cout, [&](auto ins) {
            std::cout << ":";
            std::cout << " weight=" << si.weights.at(ins);
            std::cout << " input={";
            si.get_streams_from(ins, get_inputs())([&](auto s) {
                std::cout << s << ",";
                return true;
            });
            std::cout << "}";
            if(si.has_stream(ins))
                std::cout << " stream=" << si.get_stream(ins);
        });
        std::cout << std::endl;
    }

    // No concurrency
    if(nstreams < 2)
        return;

    // Schedule instructions
    std::size_t wait_id = 0;
    std::unordered_map<instruction_ref, std::size_t> ins2wait;
    std::unordered_map<std::size_t, std::unordered_set<std::size_t>> waited_for;
    std::unordered_map<instruction_ref, std::unordered_set<std::size_t>> ins2waited;
    ins2wait.reserve(p.size());
    ins2waited.reserve(p.size());
    for(auto ins : iterator_for(p))
    {
        // Only schedule instructions that have a stream
        if(not si.has_stream(ins))
            continue;
        assert(si.weights[ins] > 0);
        // Schedule instruction on the stream
        auto stream = si.get_stream(ins);
        assert(stream < model.concurrency());
        model.sched(p, ins, stream);
        // Insert wait instructions
        if(si.is_merge_point(ins, stream))
        {
            for(auto i : si.get_recorded_instructions(ins))
            {
                if(not si.has_stream(i))
                    continue;
                auto istream = si.get_stream(i);
                if(stream == istream)
                    continue;
                // Create a new event if it hasn't been recorded
                if(not contains(ins2wait, i))
                {
                    ins2wait[i] = wait_id;
                    model.record(p, i, wait_id);
                    wait_id++;
                }
                auto w = ins2wait.at(i);
                // If we already waited for the event on this stream then dont
                // insert another wait event
                if(not contains(waited_for[stream], w))
                    model.wait(p, ins, w);
                // Store the event as waited
                waited_for[stream].insert(w);
                // Store all wait events that have been waited on prior to the recorded instruction
                waited_for[stream].insert(ins2waited[i].begin(), ins2waited[i].end());
            }
        }
        // Store wait events that have already been waited on
        if(si.is_split_point(ins, stream))
        {
            ins2waited[ins] = waited_for[stream];
        }
    }

    // Add memory conflicts
    auto conflict_table = si.get_conflicts(p);
    for(auto&& ip : conflict_table)
    {
        if(ip.second.empty())
            continue;
        std::vector<instruction_ref> args;
        args.push_back(ip.first);
        args.insert(args.end(), ip.second.begin(), ip.second.end());
        p.insert_instruction(std::next(ip.first), op::identity{}, args);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
