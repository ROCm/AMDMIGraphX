#ifndef MIGRAPHX_GUARD_OPERATORS_REVERSE_HPP
#define MIGRAPHX_GUARD_OPERATORS_REVERSE_HPP

#include <algorithm>
#include <vector>
#include <cmath>
#include <utility>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reverse
{

    int64_t axis; // 1-D, which axis will be reversed.

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "reverse"; }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        auto lens = inputs[0].lens();
        auto type = inputs[0].type();
        return shape{type, lens};
    }

    argument compute(const shape& s, std::vector<argument> args) const
    {
        argument result{s};
        auto dim_size = s.lens()[axis];
        visit_all(result, args.front())([&](auto output, auto input) {
            shape_for_each(s, [&](const auto& out_idx) {
                auto in_idx              = out_idx;
                in_idx[axis]             = dim_size - 1 - out_idx[axis];
                output[s.index(out_idx)] = input[s.index(in_idx)];
            });
        });

        return result;

    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
