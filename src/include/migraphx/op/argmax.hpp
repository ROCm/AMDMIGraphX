#ifndef MIGRAPHX_GUARD_OPERATORS_ARGMAX_HPP
#define MIGRAPHX_GUARD_OPERATORS_ARGMAX_HPP

#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/par_dfor.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct argmax
{
    int64_t axis = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "argmax"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto lens     = inputs[0].lens();
        int64_t n_dim = static_cast<int64_t>(lens.size());
        if(axis >= n_dim || axis < 0)
        {
            MIGRAPHX_THROW("ARGMAX: axis is out of range.");
        }

        lens[axis] = 1;

        return {shape::int64_type, lens};
    }

    template <class T>
    int64_t calc_argmax(T& input, std::vector<std::size_t>& indices, size_t item_num) const
    {
        auto max_val      = input(indices.begin(), indices.end());
        int64_t max_index = 0;
        for(std::size_t i = 1; i < item_num; ++i)
        {
            indices[axis] = i;
            auto cur_val  = input(indices.begin(), indices.end());
            if(max_val < cur_val)
            {
                max_val   = cur_val;
                max_index = i;
            }
        }

        return max_index;
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto batch_item_num = args.front().get_shape().lens()[axis];

        result.visit([&](auto output) {
            args[0].visit([&](auto input) {
                par_for(output_shape.elements(), [&](auto i) {
                    auto data_idx = output_shape.multi(i);
                    output[i]     = this->calc_argmax(input, data_idx, batch_item_num);
                });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
