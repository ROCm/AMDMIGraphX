#ifndef MIGRAPHX_GUARD_RTGLIB_SET_OPERATOR_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_SET_OPERATOR_IMPL_HPP

#include <utility>
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <typename Set, typename Key = typename Set::value_type>
static inline Set set_intersection(const Set& lhs, const Set& rhs)
{
    if(lhs.size() <= rhs.size())
    {
        Set iset;
        for(const Key& key : lhs)
        {
            if(rhs.count(key) > 0)
            {
                iset.insert(key);
            }
        }
        return std::move(iset);
    }
    else
    {
        return set_intersection(rhs, lhs);
    }
}

template <typename Set, typename Key = typename Set::value_type>
static inline Set set_union(const Set& lhs, const Set& rhs)
{
    Set uset{lhs};
    uset.insert(rhs.begin(), rhs.end());
    return std::move(uset);
}

template <typename Set, typename Key = typename Set::value_type>
static inline Set set_difference(const Set& lhs, const Set& rhs)
{
    Set dset{lhs};
    for(auto& iter : rhs)
    {
        dset.erase(iter);
    }
    return std::move(dset);
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
