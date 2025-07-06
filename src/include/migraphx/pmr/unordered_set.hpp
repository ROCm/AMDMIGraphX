#ifndef MIGRAPHX_GUARD_PMR_UNORDERED_SET_HPP
#define MIGRAPHX_GUARD_PMR_UNORDERED_SET_HPP

#include <migraphx/config.hpp>
#include <migraphx/pmr.hpp>

#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace pmr {
#if MIGRAPHX_HAS_PMR
template<class Key, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
using unordered_set = std::pmr::unordered_set<Key, Hash, KeyEqual>;
#else
template<class Key, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
using unordered_set = std::unordered_set<Key, Hash, KeyEqual>;
#endif

} // namespace pmr
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_PMR_UNORDERED_SET_HPP
