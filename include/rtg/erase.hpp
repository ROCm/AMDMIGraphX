#ifndef RTG_GUARD_ERASE_HPP
#define RTG_GUARD_ERASE_HPP

namespace rtg {

template <class R, class T>
auto erase(R&& r, const T& value)
{
    return r.erase(std::remove(r.begin(), r.end(), value), r.end());
}

template <class R, class P>
auto erase_if(R&& r, P&& pred)
{
    return r.erase(std::remove_if(r.begin(), r.end(), pred), r.end());
}

} // namespace rtg

#endif
