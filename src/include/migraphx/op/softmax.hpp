#ifndef MIGRAPHX_GUARD_OPERATORS_SOFTMAX_HPP
#define MIGRAPHX_GUARD_OPERATORS_SOFTMAX_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/common.hpp>
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
        value attr;
        attr["axis"] = value::array{op_normalize_attributes::include_min};
        return {{"normalize", attr}};
    }

    std::string name() const { return "softmax"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
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
