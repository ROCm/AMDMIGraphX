#ifndef MIGRAPHX_GUARD_OPERATORS_SOFTMAX_HPP
#define MIGRAPHX_GUARD_OPERATORS_SOFTMAX_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct softmax
{
    int64_t axis = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "softmax"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        int64_t n_dim = inputs[0].lens().size();
        if(axis < 0 || axis >= n_dim)
        {
            MIGRAPHX_THROW("SOFTMAX: input axis value " + std::to_string(axis) +
                           " is out of range");
        }
        return inputs.at(0);
    }

    auto output() const
    {
        return [=](auto x, auto y) { return x / y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
