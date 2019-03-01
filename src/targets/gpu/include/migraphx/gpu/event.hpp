#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_EVENT_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_EVENT_HPP

#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct create_events
{
    int num_of_events = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.num_of_events, "event"));
    }
    std::string name() const { return "gpu::create_events"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        ctx.create_events(num_of_events);
        return {};
    }
};

struct record_event
{
    int event = -1;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.event, "event"));
    }
    std::string name() const { return "gpu::record_event"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        ctx.record_event(event);
        return {};
    }

    void finalize(context& ctx, const shape&, std::vector<shape>)
    {
        assert(event >= 0);
        ctx.create_events(event);
    }
};

struct wait_event
{
    int event = -1;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.event, "event"));
    }
    std::string name() const { return "gpu::wait_event"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        ctx.wait_event(event);
        return {};
    }

    void finalize(context& ctx, const shape&, std::vector<shape>)
    {
        assert(event >= 0);
        ctx.check_events(event);
    }
};

struct set_stream
{
    int stream = -1;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.stream, "stream"));
    }
    std::string name() const { return "gpu::set_stream"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        assert(stream >= 0);
        ctx.set_stream(stream);
        return {};
    }
    void finalize(context& ctx, const shape&, const std::vector<shape>&) { ctx.set_stream(stream); }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
