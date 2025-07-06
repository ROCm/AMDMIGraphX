#ifndef MIGRAPHX_GUARD_PMR_UNORDERED_MAP_HPP
#define MIGRAPHX_GUARD_PMR_UNORDERED_MAP_HPP

#include <migraphx/config.hpp>
#include <migraphx/pmr.hpp>

#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace pmr {
#if MIGRAPHX_HAS_PMR
template<class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
using unordered_map = std::pmr::unordered_map<Key, T, Hash, KeyEqual>;
#else
template<class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
using unordered_map = std::unordered_map<Key, T, Hash, KeyEqual>;
#endif

} // namespace pmr
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_PMR_UNORDERED_MAP_HPP
