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
    bool exclusive = false, reverse = false;
    
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
        auto lens       = s.lens();

        return {s.type(), lens};
    }

    void prefix_scan(argument& arg, argument& res) const
    {
        std::vector<bool> axes(arg.get_shape().lens().size(), false);
        axes[axis] = true;
        auto& self = static_cast<const Derived&>(*this);
        bool first_pass = true;
        visit_all(arg, res)([&](auto input, auto output) { 
            shape_for_each(output.get_shape(), [&](const auto& idx) {
                if (idx[axis] == 0) {
                    output(idx.begin(), idx.end()) = input(idx.begin(), idx.end());
                }
                else {
                    auto prefix_idx = idx;
                    prefix_idx[axis] -= 1;
                    if (exclusive) {
                        output(idx.begin(), idx.end()) = input(prefix_idx.begin(), prefix_idx.end());
                        if (first_pass) {
                            output(prefix_idx.begin(), prefix_idx.end()) = 0;
                            first_pass = false;
                        }
                        output(idx.begin(), idx.end()) = self.op()(output(idx.begin(), idx.end()), output(prefix_idx.begin(), prefix_idx.end()));
                    }
                    else
                        output(idx.begin(), idx.end()) = self.op()(input(idx.begin(), idx.end()), output(prefix_idx.begin(), prefix_idx.end()));
                }
            });
        });
        /*
        visit_all(arg, res)([&](auto input, auto output) {
            std::copy(input.begin(), input.end(), output.begin());
            bool first_pass = true;
            if (reverse) {
                for (std::size_t i = lens[0] - 1; i + 1 >= std::size_t(axes[0]) + 1; --i) {
                    for (std::size_t j = lens[1] - 1; j + 1 >= std::size_t(axes[1]) + 1; --j) {
                        for (std::size_t k = lens[2] - 1; k + 1 >= std::size_t(axes[2]) + 1; --k) {
                            std::size_t idx = ((i - axes[0]) * lens[2] * lens[1]) + ((j - axes[1]) * lens[1]) + (k - axes[2]);
                            std::size_t prefix_idx = (i * lens[2] * lens[1]) + (j * lens[1]) + k;
                            if (exclusive) {
                                output[idx] = input[prefix_idx];
                                if (first_pass) {
                                    output[prefix_idx] = 0;
                                    first_pass = false;
                                }
                            }
                            output[idx] = self.op()(output[idx], output[prefix_idx]);
                        }
                    }
                }
                return;
            }
            for (std::size_t i = axes[(axis + 2) % 3]; i < std::size_t(lens[(axis + 2) % 3]); ++i) {
                for (std::size_t j = axes[(axis + 1) % 3]; j < std::size_t(lens[(axis + 1) % 3]); ++j) {
                    for (std::size_t k = axes[axis]; k < std::size_t(lens[axis]); ++k) {
                        std::size_t idx = (i * lens[axis] * lens[(axis + 1) % 3]) + (j * lens[(axis + 1) % 3]) + k;
                        std::size_t prefix_idx = ((i - axes[(axis + 2) % 3]) * lens[axis] * lens[(axis + 1) % 3]) + ((j - axes[(axis + 1) % 3]) * lens[(axis + 1) % 3]) + (k - axes[axis]);
                        if (exclusive) {
                            output[idx] = input[prefix_idx];
                            if (first_pass) {
                                output[prefix_idx] = 0;
                                first_pass = false;
                            }
                        }
                        output[idx] = self.op()(output[idx], output[prefix_idx]);
                    }
                }
            }

            for (std::size_t i = axes[0]; i < std::size_t(lens[0]); ++i) {
                for (std::size_t j = axes[1]; j < std::size_t(lens[1]); ++j) {
                    for (std::size_t k = axes[2]; k < std::size_t(lens[2]); ++k) {
                        std::size_t idx = (i * lens[2] * lens[1]) + (j * lens[1]) + k;
                        std::size_t prefix_idx = ((i - axes[0]) * lens[2] * lens[1]) + ((j - axes[1]) * lens[1]) + (k - axes[2]);
                        if (exclusive) {
                            output[idx] = input[prefix_idx];
                            if (first_pass) {
                                output[prefix_idx] = 0;
                                first_pass = false;
                            }
                        }
                        output[idx] = self.op()(output[idx], output[prefix_idx]);
                    }
                }
            }
            
        });
        */

    }
    
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        this->prefix_scan(args[0], result);

        return result; 
    }

    auto init() const {}
    prefix_scan_op() {}
    prefix_scan_op(int64_t ax) : axis(ax) {}
    prefix_scan_op(int64_t ax, bool excl) : axis(ax), exclusive(excl) {}
    prefix_scan_op(int64_t ax, bool excl, bool rev) : axis(ax), exclusive(excl), reverse(rev) {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
