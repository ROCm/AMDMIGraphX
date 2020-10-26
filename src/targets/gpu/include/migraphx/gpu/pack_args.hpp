#ifndef MIGRAPHX_GUARD_RTGLIB_PACK_ARGS_HPP
#define MIGRAPHX_GUARD_RTGLIB_PACK_ARGS_HPP

#include <migraphx/config.hpp>
#include <migraphx/requires.hpp>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct kernel_argument
{
    template <class T,
              class U = std::remove_reference_t<T>,
              MIGRAPHX_REQUIRES(not std::is_base_of<kernel_argument, T>{})>
    kernel_argument(T&& x) : size(sizeof(U)), align(alignof(U)), data(&x) // NOLINT
    {
    }
    std::size_t size;
    std::size_t align;
    void* data;
};

std::vector<char> pack_args(const std::vector<kernel_argument>& args);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
