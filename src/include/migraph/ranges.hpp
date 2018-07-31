#ifndef MIGRAPH_GUARD_MIGRAPHLIB_RANGES_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_RANGES_HPP

#include <algorithm>

namespace migraph {

template <class C, class T>
bool contains(C&& c, T&& x)
{
    return c.find(x) != c.end();
}

template <class Range, class Iterator>
void copy(Range&& r, Iterator it)
{
    std::copy(r.begin(), r.end(), it);
}

} // namespace migraph

#endif
