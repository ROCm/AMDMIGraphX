#ifndef MIGRAPHX_GUARD_RTGLIB_ALGORITHM_HPP
#define MIGRAPHX_GUARD_RTGLIB_ALGORITHM_HPP

#include <algorithm>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Iterator, class Output, class Predicate>
void group_by(Iterator start, Iterator last, Output out, Predicate pred)
{
    while(start != last)
    {
        auto it = std::partition(start, last, [&](auto&& x) { return pred(x, *start); });
        out(start, it);
        start = it;
    }
}

template <class Iterator, class Output, class Predicate>
void group_unique(Iterator start, Iterator last, Output out, Predicate pred)
{
    while(start != last)
    {
        auto it = std::find_if(start, last, [&](auto&& x) {
            return not pred(*start, x);
        });
        out(start, it);
        start = it;
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
