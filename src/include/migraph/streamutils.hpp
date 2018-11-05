#ifndef MIGRAPH_GUARD_STREAMUTILS_HPP
#define MIGRAPH_GUARD_STREAMUTILS_HPP

#include <ostream>
#include <algorithm>
#include <migraph/rank.hpp>
#include <migraph/config.hpp>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {

template <class T>
struct stream_range_container
{
    const T* r;
    stream_range_container(const T& x) : r(&x) {}

    friend std::ostream& operator<<(std::ostream& os, const stream_range_container& sr)
    {
        assert(sr.r != nullptr);
        if(!sr.r->empty())
        {
            os << sr.r->front();
            std::for_each(
                std::next(sr.r->begin()), sr.r->end(), [&](auto&& x) { os << ", " << x; });
        }
        return os;
    }
};

template <class Range>
inline stream_range_container<Range> stream_range(const Range& r)
{
    return {r};
}

namespace detail {

template <class Range>
auto stream_write_value_impl(rank<1>, std::ostream& os, const Range& r)
    -> decltype(r.begin(), r.end(), void())
{
    os << stream_range(r);
}

template <class T>
void stream_write_value_impl(rank<0>, std::ostream& os, const T& x)
{
    os << x;
}
} // namespace detail

template <class T>
void stream_write_value(std::ostream& os, const T& x)
{
    detail::stream_write_value_impl(rank<1>{}, os, x);
}

} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
