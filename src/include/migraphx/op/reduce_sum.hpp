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
    std::vector<int64_t> axes{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    std::string name() const { return "reduce_sum"; }

    std::vector<int64_t> tune_axes(std::size_t n_dim) const
    {
        auto tuned_axes = axes;
        if(tuned_axes.empty())
        {
            tuned_axes.resize(n_dim);
            std::iota(tuned_axes.begin(), tuned_axes.end(), 0);
        }
        else
        {
            for(auto& axis : tuned_axes)
            {
                int64_t s_dim = static_cast<int64_t>(n_dim);
                if(axis >= s_dim or axis < -s_dim)
                {
                    MIGRAPHX_THROW("REDUCE_MEAN: axis out of range");
                }
                if(axis < 0)
                {
                    axis += n_dim;
                }
            }
        }

        return tuned_axes;
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto s          = inputs.at(0);
        auto lens       = s.lens();
        auto tuned_axes = tune_axes(lens.size());
        for(auto axis : tuned_axes)
        {
            lens[axis] = 1;
        }

        return {s.type(), lens};
    }

    template <class T>
    void calc_sum(tensor_view<T>& input,
                  shape& batch_shape,
                  std::vector<int64_t>& tuned_axes,
                  std::vector<std::size_t>& out_idx,
                  tensor_view<T>& output) const
    {
        auto data_idx = out_idx;
        T val         = T{0};
        shape_for_each(batch_shape, [&](auto b_idx) {
            for(auto axis : tuned_axes)
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
        auto arg_lens                   = args.front().get_shape().lens();
        std::vector<int64_t> tuned_axes = tune_axes(arg_lens.size());
        std::vector<std::size_t> batch_lens(output_shape.lens().size(), 1);
        for(auto axis : tuned_axes)
        {
            batch_lens[axis] = arg_lens[axis];
        }
        shape batch_shape{output_shape.type(), batch_lens};
        visit_all(result, args[0])([&](auto output, auto input) {
            par_for(output_shape.elements(), [&](auto i) {
                auto out_idx = output_shape.multi(i);
                this->calc_sum(input, batch_shape, tuned_axes, out_idx, output);
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
