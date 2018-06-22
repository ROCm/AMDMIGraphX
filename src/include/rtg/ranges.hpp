#ifndef RTG_GUARD_RTGLIB_RANGES_HPP
#define RTG_GUARD_RTGLIB_RANGES_HPP

namespace rtg {

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

} // namespace rtg

#endif
