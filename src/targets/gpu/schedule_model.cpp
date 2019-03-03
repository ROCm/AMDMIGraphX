#include <migraphx/gpu/schedule_model.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/program.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using hip_event_ptr = MIGRAPHX_MANAGE_PTR(hipEvent_t, hipEventDestroy);

hip_event_ptr create_event()
{
    hipEvent_t event;
    auto status = hipEventCreateWithFlags(&event, hipEventDisableTiming);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to create event");
    return hip_event_ptr{event};
}

struct wait_event
{
    std::vector<std::size_t> wait_for;
    shared<hip_event_ptr> event = nullptr;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.wait_for, "wait_for"));
    }
    std::string name() const { return "gpu::wait_event"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        assert(event != nullptr);
        assert(std::none_of(wait_for.begin(), wait_for.end(), [&](auto i) {
            return i == ctx.get_current_device().stream_id();
        }));
        for(auto n : wait_for)
            ctx.get_stream(n).record(event.get());
        ctx.get_stream().wait(event.get());
        return {};
    }

    void finalize(context& ctx, const shape&, std::vector<shape>)
    {
        assert(std::none_of(wait_for.begin(), wait_for.end(), [&](auto i) {
            return i == ctx.get_current_device().stream_id();
        }));
        event = create_event();
    }
};

struct set_stream
{
    std::size_t stream = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.stream, "stream"));
    }
    std::string name() const { return "gpu::set_stream"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        ctx.set_stream(stream);
        return {};
    }
    void finalize(context& ctx, const shape&, const std::vector<shape>&) { ctx.set_stream(stream); }
};

std::size_t schedule_model::concurrency() const { return streams; }
void schedule_model::schedule_instruction(program& p, instruction_ref ins, std::size_t n) const
{
    p.insert_instruction(ins, set_stream{n});
}
void schedule_model::wait(program& p,
                          instruction_ref ins,
                          std::size_t wait_on,
                          const std::vector<std::size_t>& wait_for) const
{
    p.insert_instruction(ins, set_stream{wait_on});
    p.insert_instruction(ins, wait_event{wait_for});
}

static std::unordered_map<std::string, std::size_t> create_weight_map()
{
    return {
        {"hip::load_literal", 0},
        {"hip::allocate", 0},
        {"gpu::convolution", 4},
        {"gpu::conv_bias_relu", 4},
        {"gpu::pooling", 2},
        {"gpu::gemm", 2},
        {"gpu::concat", 1},
        {"hip::add_relu", 2},
    };
}

static const std::unordered_map<std::string, std::size_t>& weight_map()
{
    static std::unordered_map<std::string, std::size_t> m = create_weight_map();
    return m;
}

std::size_t schedule_model::weight(const operation& op) const
{
    if(weight_map().count(op.name()) == 0)
    {
        return 1;
    }
    return weight_map().at(op.name());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
