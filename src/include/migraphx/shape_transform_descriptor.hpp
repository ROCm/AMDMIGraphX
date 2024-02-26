#ifndef MIGRAPHX_GUARD_MIGRAPHX_SHAPE_TRANSFORM_DESCRIPTOR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SHAPE_TRANSFORM_DESCRIPTOR_HPP

#include <migraphx/config.hpp>
#include <cstdint>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct operation;

struct shape_transform_descriptor
{
    shape_transform_descriptor() = default;
    explicit shape_transform_descriptor(const std::vector<std::size_t>& dims);

    bool apply(const std::vector<operation>& ops);
    bool apply_reshape(const std::vector<std::size_t>& dims);
    bool apply_transpose(const std::vector<std::int64_t>& permutation);
    struct dimension
    {
        std::size_t len() const;
        struct sub
        {
            std::size_t len;
            std::vector<std::size_t> axis = {};
        };
        std::vector<sub> subdimensions;
    };
    std::vector<dimension::sub> get_all_subdimensions() const;
    std::size_t elements() const;
    std::vector<dimension> dimensions;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SHAPE_TRANSFORM_DESCRIPTOR_HPP
