#ifndef MIGRAPHX_GUARD_MIGRAPHX_RESHAPE_DIMS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_RESHAPE_DIMS_HPP

#include <migraphx/config.hpp>
#include <migraphx/optional.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct shape;

struct reshape_dims_options
{
    bool lazy = false;
};

optional<shape> reshape_dims(const shape& input,
                            const std::vector<std::size_t>& rdims, reshape_dims_options options);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_RESHAPE_DIMS_HPP


