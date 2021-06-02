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

        // auto input  = args[0].get_shape(); //float_type, {2, 16}, {16, 1}

        // std::vector<std::size_t> data;
        // args[0].visit([&](auto s) { data.assign(s.begin(), s.end()); });

        // const std::vector<std::size_t>& lens = input.lens();

        // if (axis == 0)
        // {
        //     for(std::size_t k = 0; k < lens[0]/2; k++) //4
        //     {
        //         for(std::size_t i = 0; i < lens[1]; i++) //16
        //         {
        //             std::iter_swap( data.begin() + i + (k * lens[1]), data.begin() + i +
        //             ((lens[0]-k-1)*lens[1]) );
        //         }
        //     }
        // }
        // else if (axis == 1)
        // {
        //     for (std::size_t t = 0; t < lens[0]; t++)
        //     {
        //         std::reverse( data.begin() + (t * lens[1]), data.begin() + ((t+1) * lens[1]) );
        //         //
        //     }
        // }
        // else {
        //     MIGRAPHX_THROW("reverse op can only have axis=1 or axis=0");
        // }

        // argument result{input};

        // result.visit([&](auto output) {
        //     par_for(input.elements(), [&](auto i) {
        //         output[i]     = data[i];
        //     });
        // });

        // return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
