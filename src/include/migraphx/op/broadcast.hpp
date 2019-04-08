#ifndef MIGRAPHX_GUARD_OPERATORS_BROADCAST_HPP
#define MIGRAPHX_GUARD_OPERATORS_BROADCAST_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    shape broadcast_shape;
    std::string name() const { return "broadcast"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto t     = inputs.at(0).type();
        auto input = inputs.at(0);

        std::vector<size_t> bcast_strides(broadcast_shape.lens().size(), 0);

        if(std::all_of(broadcast_shape.lens().cbegin(), broadcast_shape.lens().cend(), [&](auto x) {
               return x == 1;
           }))
        {
            if(axis != 0)
                MIGRAPHX_THROW("when broadcasting tensor of size 1, axis should be 0");
            return {t, broadcast_shape.lens(), std::move(bcast_strides)};
        }
        else
        {
            assert(broadcast_shape.lens().size() - axis >= input.lens().size());
            if(!std::equal(
                   input.lens().begin(), input.lens().end(), broadcast_shape.lens().begin() + axis))
                MIGRAPHX_THROW("when broadcasting success sizes must match");
            std::copy(input.strides().begin(), input.strides().end(), bcast_strides.begin() + axis);
            return {t, broadcast_shape.lens(), std::move(bcast_strides)};
        }
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
