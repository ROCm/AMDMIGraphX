#ifndef MIGRAPH_GUARD_MIGRAPHLIB_MAKE_SHARED_ARRAY_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_MAKE_SHARED_ARRAY_HPP

#include <memory>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

template <typename T>
std::shared_ptr<T> make_shared_array(size_t size)
{
    return std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
}

} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
