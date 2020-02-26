#ifndef MIGRAPHX_GUARD_OPERATORS_CLIP_HPP
#define MIGRAPHX_GUARD_OPERATORS_CLIP_HPP

#include <array>
#include <migraphx/op/unary.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>
#include <limits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct clip : unary<clip>
{
    float max_val = std::numeric_limits<float>::max();
    float min_val = std::numeric_limits<float>::min();

    clip() {}

    clip(float max, float min) : max_val(max), min_val(min) {}

    auto apply() const
    {
        auto max = max_val;
        auto min = min_val;
        return [max, min](auto x) {
            using type = decltype(x);
            return std::min(std::max(type(min), x), type(max));
        };
    }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.max_val, "max"), f(self.min_val, "min"));
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
