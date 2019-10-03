#include <migraphx/gpu/schedule_model.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct record_event
{
    std::size_t event = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.event, "event"));
    }
    std::string name() const { return "gpu::record_event"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        ctx.get_stream().record(ctx.get_event(event));
        return {};
    }

    void finalize(context& ctx, const shape&, const std::vector<shape>&)
    {
        ctx.create_events(event);
    }
};

struct wait_event
{
    std::size_t event = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.event, "event"));
    }
    std::string name() const { return "gpu::wait_event"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        ctx.get_stream().wait(ctx.get_event(event));
        return {};
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
void schedule_model::sched(program& p, instruction_ref ins, std::size_t n) const
{
    auto last_stream = std::find_if(std::make_reverse_iterator(ins),
                                    std::make_reverse_iterator(p.begin()),
                                    [&](auto&& i) { return i.name() == "gpu::set_stream"; });
    if(last_stream != std::make_reverse_iterator(p.begin()))
    {
        auto&& op = any_cast<set_stream>(last_stream->get_operator());
        // If the same stream was set earlier then skip
        if(op.stream == n)
            return;
    }
    p.insert_instruction(ins, set_stream{n});
}

void schedule_model::wait(program& p, instruction_ref ins, std::size_t wait_id) const
{
    p.insert_instruction(ins, wait_event{wait_id});
}
void schedule_model::record(program& p, instruction_ref ins, std::size_t wait_id) const
{
    p.insert_instruction(std::next(ins), record_event{wait_id});
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
        {"gpu::concat", 1}
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
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
