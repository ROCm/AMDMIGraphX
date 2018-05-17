#ifndef RTG_GUARD_STREAMUTILS_HPP
#define RTG_GUARD_STREAMUTILS_HPP

#include <ostream>
#include <algorithm>

namespace rtg {

template<class T>
struct stream_range_container
{
    const T* r;
    stream_range_container(const T& x)
    : r(&x)
    {}

    friend std::ostream& operator<<(std::ostream& os, const stream_range_container& sr)
    {
        assert(sr.r != nullptr);
        if(!sr.r->empty())
        {
            os << sr.r->front();
            std::for_each(std::next(sr.r->begin()), sr.r->end(), [&](auto&& x) { os << ", " << x; });
        }
        return os;
    }
};

template <class Range>
inline stream_range_container<Range> stream_range(const Range& r)
{
    return {r};
}

} // namespace rtg

#endif
