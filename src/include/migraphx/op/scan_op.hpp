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
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

template <class Derived>
struct scan_op : op_name<Derived>
{
    std::vector<int64_t> axis{};
    
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
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
        auto s          = inputs.at(0);
        auto lens       = s.lens();

        return {s.type(), lens};
    }
    
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        std::vector<bool> axes(3, false);
        axes[axis[0]] = true;
        auto lens = output_shape.lens();
        std::size_t nelements = args[0].get_shape().elements();
        visit_all(result, args[0])([&](auto output, auto input) {
            for (std::size_t i = 0; i < nelements; ++i)
                output[i] = input[i];
            for (std::size_t i = axes[0]; i < lens[0]; ++i) {
                for (std::size_t j = axes[1]; j < lens[1]; ++j) {
                    for (std::size_t k = axes[2]; k < lens[2]; ++k) {
                        output[(i * lens[2] * lens[1]) + (j * lens[1]) + k] += 
                            output[((i - axes[0]) * lens[2] * lens[1]) + ((j - axes[1]) * lens[1]) + (k - axes[2])];
                    }
                }
            }
        });

        return result; 
    }

    auto init() const {}
    scan_op() {}
    scan_op(std::vector<int64_t> ax) : axis(std::move(ax)) {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
