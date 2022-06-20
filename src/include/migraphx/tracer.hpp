#ifndef MIGRAPHX_GUARD_RTGLIB_TRACER_HPP
#define MIGRAPHX_GUARD_RTGLIB_TRACER_HPP

#include <ostream>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct tracer
{
    tracer() {}

    tracer(std::ostream& s) : os(&s) {}

    bool enabled() const { return os != nullptr; }

    template <class... Ts>
    void operator()(const Ts&... xs) const
    {
        if(os != nullptr)
        {
            swallow{*os << xs...};
            *os << std::endl;
        }
    }

    private:
    std::ostream* os = nullptr;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
