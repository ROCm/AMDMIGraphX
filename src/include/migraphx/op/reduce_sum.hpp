#ifndef MIGRAPHX_GUARD_OPERATORS_SUM_HPP
#define MIGRAPHX_GUARD_OPERATORS_SUM_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
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
        auto s    = inputs.at(0);
        auto lens = s.lens();
        for(auto axis : axes)
            lens[axis] = 1;
        return {s.type(), lens};
    }

    template <class T>
    void calc_sum(tensor_view<T>& input,
                  shape& batch_shape,
                  std::vector<std::size_t>& out_idx,
                  tensor_view<T>& output) const
    {
        auto data_idx = out_idx;
        T val         = T{0};
        shape_for_each(batch_shape, [&](auto b_idx) {
            for(auto axis : axes)
            {
                data_idx[axis] = b_idx[axis];
            }
            val += input(data_idx.begin(), data_idx.end());
        });

        output(out_idx.begin(), out_idx.end()) = val;
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto arg_lens = args.front().get_shape().lens();
        std::vector<std::size_t> batch_lens(output_shape.lens().size(), 1);
        for(auto axis : axes)
        {
            batch_lens[axis] = arg_lens[axis];
        }
        shape batch_shape{output_shape.type(), batch_lens};
        visit_all(result, args[0])([&](auto output, auto input) {
            par_for(output_shape.elements(), [&](auto i) {
                auto out_idx = output_shape.multi(i);
                this->calc_sum(input, batch_shape, out_idx, output);
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
