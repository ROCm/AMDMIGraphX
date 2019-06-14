#ifndef MIGRAPHX_GUARD_OPERATORS_LRN_HPP
#define MIGRAPHX_GUARD_OPERATORS_LRN_HPP

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

struct lrn
{
    float alpha = 0.0001;
    float beta  = 0.75;
    float bias  = 1.0;
    int size    = 1;
    std::string name() const { return "lrn"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"),
                    f(self.beta, "beta"),
                    f(self.bias, "bias"),
                    f(self.size, "size"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return inputs.front();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
