#ifndef MIGRAPH_GUARD_RTGLIB_SET_OPERATOR_IMPL_HPP
#define MIGRAPH_GUARD_RTGLIB_SET_OPERATOR_IMPL_HPP

namespace migraphx {
namespace set_op {
    
template <typename Set, typename Key = typename Set::value_type>
static inline Set
set_intersection(const Set& lhs, const Set& rhs) {
    if (lhs.size() <= rhs.size()) {
        Set iset;
        for (const Key& key : lhs) {
            if (rhs.count(key) > 0) {
                iset.insert(key);
            }
        }
        return std::move(iset);
    } else {
        return set_intersection(rhs, lhs);
    }
}

template <typename Set, typename Key = typename Set::value_type>
static inline Set
set_union(const Set& lhs, const Set& rhs) {
    Set uset{lhs};
    uset.insert(rhs.begin(), rhs.end());
    return std::move(uset);
}

template <typename Set, typename Key = typename Set::value_type>
static inline Set
set_difference(const Set& lhs, const Set& rhs) {
    Set dset{lhs};
    for (auto& iter : rhs) {
        dset.erase(iter);
    }
    return std::move(dset);
}
    
} // namespace set_op
} // namespace migraphx

#endif
