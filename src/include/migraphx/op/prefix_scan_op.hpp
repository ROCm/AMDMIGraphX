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
    bool reverse = false;
    
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"), 
                    f(self.exclusive, "exclusive"), 
                    f(self.reverse, "reverse"));
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

        return {s.type(), s.lens()};
    }

    void prefix_scan(argument& arg, argument& res) const
    {
        auto& self = static_cast<const Derived&>(*this);
        if (reverse) {
            visit_all(arg, res)([&](auto input, auto output) { 
                shape_for_each_reverse(output.get_shape(), [&](const auto& idx) {
                    if (idx[axis] == arg.get_shape().lens()[axis] - 1) {
                        if (exclusive) 
                            output(idx.begin(), idx.end()) = 0;
                        else
                            output(idx.begin(), idx.end()) = input(idx.begin(), idx.end());
                    }
                    else {
                        auto prefix_idx = idx;
                        prefix_idx[axis] += 1;
                        if (exclusive) 
                            output(idx.begin(), idx.end()) = self.op()(input(prefix_idx.begin(), prefix_idx.end()), output(prefix_idx.begin(), prefix_idx.end()));
                        else
                            output(idx.begin(), idx.end()) = self.op()(input(idx.begin(), idx.end()), output(prefix_idx.begin(), prefix_idx.end()));
                    }
                });
            });
        }
        else {
            visit_all(arg, res)([&](auto input, auto output) { 
                shape_for_each(output.get_shape(), [&](const auto& idx) {
                    if (idx[axis] == 0) {
                        if (exclusive) 
                            output(idx.begin(), idx.end()) = 0;
                        else
                            output(idx.begin(), idx.end()) = input(idx.begin(), idx.end());
                    }
                    else {
                        auto prefix_idx = idx;
                        prefix_idx[axis] -= 1;
                        if (exclusive) 
                            output(idx.begin(), idx.end()) = self.op()(input(prefix_idx.begin(), prefix_idx.end()), output(prefix_idx.begin(), prefix_idx.end()));
                        else
                            output(idx.begin(), idx.end()) = self.op()(input(idx.begin(), idx.end()), output(prefix_idx.begin(), prefix_idx.end()));
                    }
                });
            });
        }
    }
    
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        this->prefix_scan(args[0], result);

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
