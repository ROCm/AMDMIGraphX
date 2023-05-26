#ifndef MIGRAPHX_GUARD_MIGRAPHX_HASH_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_HASH_HPP

#include <migraphx/config.hpp>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
std::size_t hash_value(const T& v)
{
    return std::hash<T>{}(v);
}

template <class T>
void hash_combine(std::size_t& seed, const T& v)
{
    seed ^= hash_value(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_HASH_HPP
