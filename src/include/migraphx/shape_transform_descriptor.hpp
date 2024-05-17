#ifndef MIGRAPHX_GUARD_MIGRAPHX_SHAPE_TRANSFORM_DESCRIPTOR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SHAPE_TRANSFORM_DESCRIPTOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/optional.hpp>
#include <cstdint>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct operation;

// The shape_transform_descriptor class is data structure to simplify shape
// transformations like reshape, transpose, broadcast, etc. This is made up
// of a collection of dimensions which are a collection of subdimensions.
//
// Each subdimension has an axis and a `len`. The `len` is the length of the
// subdimension. The axis represents the axis the dimension originated. It is
// represented as a vector, the first element represents the axis in the
// original dimension and the additional elements are used when such
// dimension is split. The axis is empty when its a broadcasted dimension,
// and a hidden axis can be set if the dimension is associated with a `1`
// dimension in the original shape.
//
// This will first record shape transformations with the `apply` method. This
// will manipulate this data structure to represent how the transformation
// changes the dimensions.
//
// For example, if we start with an initial dimensions as `[x, y, z]` then
// each dimension will have one subdimension that corresponds to each
// original dimension: `[[x:0]], [[y:1]], [[z:2]]`.
//
// When a transpose is applied we would just permutate the dimensions.
//
// When a reshape that would merge dimensions together then the subdimensions
// are copied to the same subdimension. So if we reshape the dimensions as `
// [x*y, z]` then it would become `[[x:0], [y:1]], [[z:1]]`. If the reshape
// splits the dimension then the subdimension is copied to each dimension and
// the axis is updated to maintain the order. So a reshape of `[2, x/2, y,
// z]` would become: `[[2:0,0]], [[x/2:0,1]], [[y:1]], [[z:2]]`.
//
// After recording the operators, `simplify` method is used to simplify the
// data structure such as merging adjacent dimension, etc. The `generate`
// method is called to generate the operators need to do this
// transformation.
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
            optional<std::size_t> hidden_axis = nullopt;
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
