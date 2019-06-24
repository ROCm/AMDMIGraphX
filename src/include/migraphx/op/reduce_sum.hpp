#ifndef MIGRAPHX_GUARD_OPERATORS_SUM_HPP
#define MIGRAPHX_GUARD_OPERATORS_SUM_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_sum
{
    std::vector<std::size_t> axes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    std::string name() const { return "reduce_sum"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto s = inputs.at(0);
        auto lens = s.lens();
        for(auto axis:axes)
            lens[axis] = 1;
        return {s.type(), lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input){
            shape_for_each(input.get_shape(), [&](auto&& in_idx) {
                auto out_idx = in_idx;
                for(auto axis:axes)
                    out_idx[axis] = 0;
                output(out_idx.begin(), out_idx.end()) += input(in_idx.begin(), in_idx.end());
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
