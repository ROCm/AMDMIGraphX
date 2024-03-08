#ifndef MIGRAPHX_GUARD_MIGRAPHX_SHAPE_TRANSFORM_DESCRIPTOR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SHAPE_TRANSFORM_DESCRIPTOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/optional.hpp>
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
    bool apply_broadcast(const std::vector<std::size_t>& out_lens,
                         optional<std::size_t> axis = nullopt);
    void simplify();
    std::size_t elements() const;
    std::vector<operation> generate() const;

    struct dimension
    {
        void simplify();
        std::size_t len() const;
        struct sub
        {
            std::size_t len;
            std::vector<std::size_t> axis = {};
            optional<std::size_t> hidden_axis   = nullopt;
        };
        std::vector<sub> subdimensions;
    };
    std::vector<dimension> dimensions;
    std::size_t rank = 0;
};

std::vector<operation> optimize_shape_transforms(const std::vector<std::size_t>& dims,
                                                 const std::vector<operation>& ops);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SHAPE_TRANSFORM_DESCRIPTOR_HPP
