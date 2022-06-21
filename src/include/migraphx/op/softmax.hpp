#ifndef MIGRAPHX_GUARD_OPERATORS_SOFTMAX_HPP
#define MIGRAPHX_GUARD_OPERATORS_SOFTMAX_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
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

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "softmax"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        if(inputs.at(0).packed())
        {
            return inputs.at(0);
        }
        else
        {
            auto lens = inputs.at(0).lens();
            return {inputs.at(0).type(), lens};
        }
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
