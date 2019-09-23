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
#include <set>
#include <deque>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

auto get_inputs()
{
    return [](auto i) { return i->inputs(); };
}

auto get_outputs()
{
    return [](auto i) { return i->outputs(); };
}

struct dominator_info
{
    bool strictly_dominate(instruction_ref ins1, instruction_ref ins2)
    {
        if(ins1 != ins2)
        {
            auto iter = ins2;
            while(contains(ins2idom, iter))
            {
                if(ins1 == ins2idom[iter])
                    return true;
                iter = ins2idom[iter];
            }
        }
        return false;
    }

    void compute_dominator(program& p, std::unordered_map<instruction_ref, std::size_t>& ins2stream)
    {
        std::size_t num_ins = p.size();
        if(num_ins == 0)
        {
            return;
        }

        std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>> ins2dominators;
        auto& ins_dom_tree = ins2idom;

        for(auto ins : reverse_iterator_for(p))
        {
            // if(!contains(ins2stream, ins))
            // {
            //     continue;
            // }

            instruction_ref ins_tmp = p.end();

            int output_num = 0;
            // find dominators
            for(auto& output : ins->outputs())
            {
                // if(!contains(ins2stream, output))
                // {
                //     continue;
                // }

                output_num++;
                if(ins_tmp == p.end())
                {
                    ins2dominators[ins] = ins2dominators[output];
                }
                else
                {
                    std::unordered_set<instruction_ref> dom_set;
                    for(auto& it : ins2dominators[ins])
                    {
                        if(contains(ins2dominators[output], it))
                        {
                            dom_set.insert(it);
                        }
                    }
                    ins2dominators[ins] = dom_set;
                }
                ins_tmp = output;
            }

            if(output_num == 1)
            {
                ins_dom_tree[ins] = ins_tmp;
            }
            else if(output_num > 0)
            {
                find_dominator_tree(ins2dominators, ins, ins2idom, ins2idom);
            }
        }
    }

    void find_dominator_tree(
        std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>>& ins2dominators,
        instruction_ref ins,
        std::unordered_map<instruction_ref, instruction_ref>& ins_dom_tree,
        std::unordered_map<instruction_ref, instruction_ref>& idom)
    {
        for(auto& iter1 : ins2dominators[ins])
        {
            auto dom_check = [& dom_tree = idom, ins1 = iter1 ](instruction_ref ins2)
            {
                if(ins1 == ins2)
                    return false;
                auto iter = ins2;
                while(contains(dom_tree, iter))
                {
                    if(ins1 == dom_tree[iter])
                        return true;
                    iter = dom_tree[iter];
                }
                return false;
            };

            // check whether iter1 strictly dominates or post-dominates any other notes in
            // p_ins's dominators or post-dominators.
            if(!std::any_of(ins2dominators[ins].begin(), ins2dominators[ins].end(), dom_check))
            {
                assert(!contains(ins_dom_tree, ins));
                ins_dom_tree[ins] = iter1;
            }
        }
    }

    std::unordered_map<instruction_ref, instruction_ref> ins2idom;
};

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

    std::vector<instruction_ref>::iterator sort_args(std::vector<instruction_ref>& args)
    {
        if(args.size() < 2)
        {
            return args.end();
        }

        const std::size_t min_partition_threshold = 2;
        auto compare                              = by(std::greater<>{}, [&](auto x) {
            return std::make_tuple(this->weights[x], x->inputs().size());
        });
        std::sort(args.begin(), args.end(), compare);

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

    void assign_streams(program& p, std::size_t n)
    {
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
        std::vector<std::size_t> streams(n - 1);
        // Assign streams for the other partitions
        for(auto&& ins_part : partitions)
        {
            std::sort(
                ins_part.second.begin(), ins_part.second.end(), by(std::greater<>{}, [](auto&& x) {
                    return std::make_tuple(x.weight, x.instructions.size());
                }));
            for(auto&& part : ins_part.second)
            {
                auto stream = std::min_element(streams.begin(), streams.end()) - streams.begin();
                set_stream(part, stream + 1);
                streams[stream] += part.weight;
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
        std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>> merge_to;
        dominator_info di;
        di.compute_dominator(p, ins2stream);

        result.reserve(p.size());
        merge_to.reserve(p.size());
        for(auto ins : reverse_iterator_for(p))
        {
            for(auto&& output : ins->outputs())
            {
                if(is_merge_point(output))
                    merge_to[ins].insert(output);
                merge_to[ins].insert(merge_to[output].begin(), merge_to[output].end());
            }

            assert(merge_to.find(ins) != merge_to.end());
            std::unordered_set<instruction_ref> del_set;
            for(auto merge : merge_to[ins])
            {
                if(di.strictly_dominate(merge, ins))
                {
                    del_set.insert(merge);
                }
            }

            std::cout << "del_set size = " << del_set.size() << std::endl;
            for(auto del_ins : del_set)
            {
                merge_to[ins].erase(del_ins);
            }

            auto streams = this->get_streams(ins);
            // Collect concur instructions for each merge point.
            for(auto& merge : merge_to[ins])
            {
                for(auto stream : streams)
                {
                    if(result[merge].size() <= stream)
                        result[merge].resize(stream + 1);
                    auto&& r = result[merge][stream];
                    r.push_back(ins);
                    // Copy inputs if they dont have a has_streamontext
                    // free). Inputs without a stream chas_stream
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
        std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>> conflict_table;
        auto concur_ins = this->find_concurrent_instructions(p);
        for(auto&& merge : concur_ins)
        {
            std::cout << "size = " << merge.second.size() << std::endl;
            for(auto st : merge.second)
            {
                std::cout << "\tsub_size = " << st.size() << std::endl;
            }
            conflict_table[merge.first].reserve(concur_ins.size() * 2);
        }

        for(auto&& merge : concur_ins)
        {
            std::cout << "ins_name = " << merge.first->name() << std::endl;
            std::cout << "size = " << merge.second.size() << std::endl;
            for(auto st : merge.second)
            {
                std::cout << "\tsub_size = " << st.size() << std::endl;
            }
            dfor(merge.second.size(), merge.second.size())([&](auto i, auto j) {
                if(i == j)
                    return;
                for(auto ins1 : merge.second[i])
                {
                    auto p1 = std::distance(ins1, merge.first);
                    for(auto ins2 : merge.second[j])
                    {
                        if(ins1 == ins2)
                            continue;
                        auto p2 = std::distance(ins2, merge.first);
                        // The smaller distance means the instruction occurs later
                        if(p1 > p2)
                            conflict_table[ins2].insert(ins1);
                        else
                            conflict_table[ins1].insert(ins2);
                    }
                }
            });
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
    si.assign_streams(p, model.concurrency());

    if(enabled(MIGRAPHX_TRACE_COMPILE{}))
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
