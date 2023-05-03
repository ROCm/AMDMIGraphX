#ifndef MIGRAPHX_GUARD_MIGRAPHX_COMMON_DIMS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_COMMON_DIMS_HPP

#include <migraphx/config.hpp>
#include <cstdint>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct common_dims
{
    static common_dims compute(const std::vector<std::size_t>& dims1,
                               const std::vector<std::size_t>& dims2);
    bool empty() const
    {
        return dims.empty();
    }
    std::vector<std::size_t> dims;
    std::vector<std::vector<std::size_t>> axes_map1;
    std::vector<std::vector<std::size_t>> axes_map2;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_COMMON_DIMS_HPP
