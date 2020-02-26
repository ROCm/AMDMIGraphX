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

struct clip
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

    std::string name() const { return "clip"; }


    shape compute_shape(std::vector<shape> inputs) const
    {
        auto s = inputs.at(0);
        if(s.packed())
        {
            return s;
        }
        else
        {
            return {s.type(), s.lens()};
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto in_shape = args[0].get_shape();
        if(in_shape.packed())
        {
            shape std_in_shape{in_shape.type(), in_shape.lens()};
            shape std_out_shape{output_shape.type(), output_shape.lens()};
            argument arg_in{std_in_shape, args[0].data()};
            argument arg_out{std_out_shape, result.data()};
            arg_out.visit([&](auto output) {
                arg_in.visit([&](auto input) {
                    std::transform(input.begin(),
                                   input.end(),
                                   output.begin(),
                                   apply());

                });
            });
        }
        else
        {
            result.visit([&](auto output) {
                args[0].visit([&](auto input) {
                    shape_for_each(output.get_shape(), [&](const auto& idx) {
                        output(idx.begin(), idx.end()) = apply()(
                            input(idx.begin(), idx.end()));
                    });
                });
            });
        }

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
