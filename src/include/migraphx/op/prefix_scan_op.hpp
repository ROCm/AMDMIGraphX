#ifndef MIGRAPHX_GUARD_OPERATORS_SCAN_OP_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCAN_OP_HPP

#include <migraphx/op/name.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

template <class Derived>
struct prefix_scan_op : op_name<Derived>
{
    int64_t axis;
    bool exclusive = false;
    bool reverse   = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.axis, "axis"), f(self.exclusive, "exclusive"), f(self.reverse, "reverse"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto s = inputs.front();
        if(s.broadcasted())
        {
            return {s.type(), s.lens()};
        }
        else
        {
            return s.with_lens(s.lens());
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto s = args[0].get_shape();
        if(s == output_shape)
        {
            result = args[0].copy();
        }
        else
        {
            visit_all(result, args[0])([&](auto output, auto input) {
                par_for(output_shape.elements(),
                        [&](auto i) { output[output_shape.index(i)] = input[s.index(i)]; });
            });
            s = output_shape;
        }
        auto slice = shape{s.type(), {s.lens()[axis]}, {s.strides()[axis]}};
        auto lens  = s.lens();
        lens[axis] = 1;
        auto batch = shape{s.type(), lens, s.strides()};
        auto& self = static_cast<const Derived&>(*this);
        result.visit([&](auto output) {
            using type = decltype(output);
            par_for(batch.elements(), [&](auto i) {
                auto* start = output.data() + batch.index(i);
                type x{slice, start};
                if(reverse)
                {
                    if(exclusive)
                    {
                        std::copy(++x.begin(), x.end(), x.begin());
                        x.back() = 0;
                    }
                    std::partial_sum(std::make_reverse_iterator(x.end()),
                                     std::make_reverse_iterator(x.begin()),
                                     std::make_reverse_iterator(x.end()),
                                     self.op());
                }
                else
                {
                    if(exclusive)
                    {
                        std::copy_backward(x.begin(), --x.end(), x.end());
                        x.front() = 0;
                    }
                    std::partial_sum(x.begin(), x.end(), x.begin(), self.op());
                }
            });
        });

        return result;
    }

    auto init() const {}
    prefix_scan_op() : axis(0) {}
    prefix_scan_op(int64_t ax) : axis(ax) {}
    prefix_scan_op(int64_t ax, bool excl) : axis(ax), exclusive(excl) {}
    prefix_scan_op(int64_t ax, bool excl, bool rev) : axis(ax), exclusive(excl), reverse(rev) {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
