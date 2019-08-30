#ifndef MIGRAPHX_GUARD_OPERATORS_OP_HPP
#define MIGRAPHX_GUARD_OPERATORS_OP_HPP

//#include <migraphx/half.hpp>
#include <migraphx/op/name.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct lowest
{
    template <class T>
    operator T() const
    {
        return std::numeric_limits<T>::lowest();
    }
};

struct highest
{
    template <class T>
    operator T() const
    {
        return std::numeric_limits<T>::max();
    }
};

struct zero
{
    template <class T>
    operator T() const
    {
        return T{0};
    }
};

struct sum_op
{
    std::size_t elem_num = 1;
    template <class T>
    T operator()(T x, T y) const
    {
        return x + y;
    }

    auto init() const { return zero(); }

    auto output() const
    {
        return [=](auto val) { return val; };
    }
};

struct mean_op
{
    std::size_t elem_num = 1;
    template <class T>
    T operator()(T x, T y) const
    {
        return x + y;
    }

    auto init() const { return zero(); }

    auto output() const
    {
        return [=](auto val) { return val / elem_num; };
    }
};

template <class Derived, class Op>
struct reduce_op : op_name<Derived>
{
    std::vector<std::int64_t> axes{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

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
                    MIGRAPHX_THROW("REDUCE_OP: axis out of range");
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
    void tune_dims(const std::vector<int64_t>& tuned_axes,
                   const std::vector<T>& in_lens,
                   std::vector<T>& out_lens) const
    {
        for(auto axis : tuned_axes)
        {
            out_lens[axis] = in_lens[axis];
        }
    }

    template <class T>
    void reduce(tensor_view<T>& input,
                shape& batch_shape,
                std::vector<int64_t>& tuned_axes,
                std::vector<std::size_t>& out_idx,
                tensor_view<T>& output,
                Op op) const
    {
        auto data_idx = out_idx;
        T val         = op.init();
        shape_for_each(batch_shape, [&](auto b_idx) {
            tune_dims(tuned_axes, b_idx, data_idx);
            val = op(input(data_idx.begin(), data_idx.end()), val);
        });

        output(out_idx.begin(), out_idx.end()) = op.output()(val);
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto arg_lens   = args.front().get_shape().lens();
        auto tuned_axes = tune_axes(arg_lens.size());
        std::vector<std::size_t> batch_lens(output_shape.lens().size(), 1);
        tune_dims(tuned_axes, arg_lens, batch_lens);
        shape batch_shape{output_shape.type(), batch_lens};
        visit_all(result, args[0])([&](auto output, auto input) {
            par_for(output_shape.elements(), [&](auto i) {
                auto out_idx = output_shape.multi(i);
                this->reduce(
                    input, batch_shape, tuned_axes, out_idx, output, Op{batch_shape.elements()});
            });
        });

        return result;
    }

    reduce_op() {}
    reduce_op(std::vector<int64_t> ax) : axes(std::move(ax)) {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
