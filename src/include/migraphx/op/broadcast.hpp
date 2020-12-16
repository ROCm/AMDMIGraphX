#ifndef MIGRAPHX_GUARD_OPERATORS_BROADCAST_HPP
#define MIGRAPHX_GUARD_OPERATORS_BROADCAST_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/// The broadcast operator performs the numpy-style broadcasting of an axis of a given tensor. This
/// is achieved primarily by setting the stride of the broadcasted axis to zero. Linear indicies are
/// computed from multi-indicies by computing the inner product on the multi-index with the strides.
/// For example, if we have a tensor A(2,3) it has lengths of (2,3) and strides of (3,1). If we want
/// to compute the linear offset that corresponds to the element on the 2nd row (i = 1) and 3rd
/// column (j = 2), we compute the following inner product (1,2) dot (3, 1) = 1*3 + 2*1 = 5. It is
/// obvious from there that we can negate the effects of a given axis by setting the stride of that
/// axis to zero.
struct broadcast
{
    uint64_t axis = 0;
    std::vector<std::size_t> broadcast_lens;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"), f(self.broadcast_lens, "dims"));
    }

    std::string name() const { return "broadcast"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto input = inputs.at(0);
        auto t     = input.type();

        std::vector<size_t> bcast_strides(broadcast_lens.size(), 0);
        // the broacast op is deprecated now, so not handling the negative
        // value of axis anymore
        if(axis >= broadcast_lens.size())
        {
            MIGRAPHX_THROW("BROADCAST : axis is out of range");
        }

        if(broadcast_lens.size() - axis < input.lens().size())
        {
            MIGRAPHX_THROW("BROADCAST: (broadcast ndims - axis) is less than input ndims");
        }

        if(!std::equal(input.lens().begin(), input.lens().end(), broadcast_lens.begin() + axis))
        {
            MIGRAPHX_THROW("BROADCAST: when broadcasting, succeeding sizes must match");
        }
        std::copy(input.strides().begin(), input.strides().end(), bcast_strides.begin() + axis);

        shape output{t, broadcast_lens, std::move(bcast_strides)};
        if(output.elements() < input.elements())
            MIGRAPHX_THROW("BROADCAST: output size must be greater than or equal to input size");
        return output;
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
