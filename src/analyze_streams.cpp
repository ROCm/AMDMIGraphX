#include <migraphx/analyze_streams.hpp>
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool happens_before(const std::vector<std::size_t>& e1, const std::vector<std::size_t>& e2)
{
    return std::equal(e1.begin(), e1.end(), e2.begin(), e2.end(), std::less_equal<>{}) and not std::equal(e1.begin(), e1.end(), e2.begin(), e2.end(), std::greater_equal<>{});
}

std::vector<stream_race> analyze_streams(const program& p, const stream_model& m)
{
    using vector_clock = std::vector<std::size_t>;
    std::vector<stream_race> races;
    auto nstream = m.get_nstream();
    std::vector<vector_clock> vclock(nstream+1, vector_clock(nstream+1));
    std::unordered_map<instruction_ref, vector_clock> timestamp;
    std::unordered_map<std::size_t,vector_clock> events;
    for(auto ins:iterator_for(p))
    {
        if (not m.has_stream(ins))
            continue;
        std::size_t s = m.get_stream(ins);
        if (m.is_record(ins))
        {
            vclock[s][s]++;
            auto event = m.get_event_id(ins);
            events[event] = vclock[s];
        }
        else if (m.is_wait(ins))
        {
            auto event = m.get_event_id(ins);
            auto payload = events[event];
            std::transform(vclock[s].begin(), vclock[s].end(), payload.begin(), vclock[s].begin(), [&](auto x, auto y) {
                return std::max(x, y);
            });
            vclock[s][s]++;
        }
        else
        {
            vclock[s][s]++;
        }
        timestamp[ins] = vclock[s];
    }
    for(auto ins:iterator_for(p))
    {
        if (not m.has_stream(ins))
            continue;
        if (ins->inputs().empty())
            continue;
        std::size_t s = m.get_stream(ins);
        // Find inputs from different streams
        std::vector<instruction_ref> inputs;
        fix([&](auto self, auto start) {
            for(auto input:start->inputs())
            {
                if (not m.has_stream(input))
                    self(input);
                else if (m.get_stream(input) != s)
                    inputs.push_back(input);
            }
        })(ins);
        auto it = std::find_if(inputs.begin(), inputs.end(), [&](auto input) {
            return not happens_before(timestamp[input], timestamp[ins]);
        });
        if (it != inputs.end())
        {
            races.push_back({ins, *it});
        }
    }

    return races;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
