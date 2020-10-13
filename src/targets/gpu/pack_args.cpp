#include <migraphx/gpu/pack_args.hpp>
#include <migraphx/requires.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

template <typename T, MIGRAPHX_REQUIRES(std::is_integral<T>{})>
inline T round_up_to_next_multiple_nonnegative(T x, T y)
{
    T tmp = x + y - 1;
    return tmp - tmp % y;
}

std::vector<char> pack_args(const std::vector<std::pair<std::size_t, void*>>& args)
{
    std::vector<char> kernargs;
    for(auto&& arg : args)
    {
        std::size_t n = arg.first;
        const auto* p = static_cast<const char*>(arg.second);
        // Insert padding
        // std::size_t alignment    = arg.first;
        // std::size_t padding      = (alignment - (prev % alignment)) % alignment;
        kernargs.insert(kernargs.end(),
                        round_up_to_next_multiple_nonnegative(kernargs.size(), n) - kernargs.size(),
                        0);
        kernargs.insert(kernargs.end(), p, p + n);
    }
    return kernargs;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
