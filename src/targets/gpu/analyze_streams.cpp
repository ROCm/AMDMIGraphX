#include <migraphx/gpu/analyze_streams.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_stream_model
{
    std::size_t max_stream = 0;
    std::unordered_map<migraphx::instruction_ref, std::size_t> ins2stream{};
    std::size_t get_nstream() const { return max_stream + 1; }
    std::size_t get_stream(migraphx::instruction_ref ins) const { return ins2stream.at(ins); }
    std::size_t get_event_id(migraphx::instruction_ref ins) const
    {
        auto v = ins->get_operator().to_value();
        return v["event"].to<std::size_t>();
    }
    bool has_stream(migraphx::instruction_ref ins) const { return ins2stream.count(ins) > 0; }
    bool is_record(migraphx::instruction_ref ins) const
    {
        return ins->name() == "gpu::record_event";
    }
    bool is_wait(migraphx::instruction_ref ins) const { return ins->name() == "gpu::wait_event"; }
};

stream_model make_stream_model(const program& p)
{
    hip_stream_model m;
    std::size_t stream = 0;
    for(auto ins : iterator_for(p))
    {
        if(ins->name() == "gpu::set_stream")
        {
            auto v       = ins->get_operator().to_value();
            stream       = v["stream"].to<std::size_t>();
            m.max_stream = std::max(stream, m.max_stream);
        }
        if(ins->get_operator().is_context_free())
            continue;
        if(contains({"hip::hip_allocate_memory", "hip::hip_copy_literal", "@param"}, ins->name()))
            continue;
        m.ins2stream[ins] = stream;
    }
    return m;
}

std::vector<stream_race> analyze_streams(const program& p)
{
    return migraphx::analyze_streams(p, make_stream_model(p));
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
