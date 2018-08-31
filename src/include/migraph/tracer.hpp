#ifndef MIGRAPH_GUARD_RTGLIB_TRACER_HPP
#define MIGRAPH_GUARD_RTGLIB_TRACER_HPP

#include <ostream>
#include <migraph/functional.hpp>

namespace migraph {

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

} // namespace migraph

#endif
