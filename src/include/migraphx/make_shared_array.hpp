#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_MAKE_SHARED_ARRAY_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_MAKE_SHARED_ARRAY_HPP

#include <memory>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <typename T>
std::shared_ptr<T> make_shared_array(size_t size)
{
    return std::shared_ptr<T>(new T[size](), std::default_delete<T[]>()); // NOLINT
}

template <class T, class Iterator>
std::shared_ptr<T> make_shared_array(Iterator start, Iterator last)
{
    auto result = make_shared_array<T>(std::distance(start, last));
    std::copy(start, last, result.get());
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
