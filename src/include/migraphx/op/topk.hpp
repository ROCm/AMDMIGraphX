#ifndef MIGRAPHX_GUARD_OPERATORS_GATHER_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHER_HPP

#include <algorithm>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape_for_each.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct topk
{
    int64_t k;
    int64_t axis = 0;
    bool largest = true;
    bool sorted  = true;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.k, "topk"),
                    f(self.axis, "axis"),
                    f(self.largest, "largest"),
                    f(self.sorted, "sorted"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "topk"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto lens = inputs.at(0).lens();
        auto type = inputs.at(0).type();

        lens[axis] = k;

        shape s_val{type, lens};
        shape s_ind{shape::int64_type, lens};

        return shape({s_val, s_ind});
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto vec_ss = output_shape.sub_shapes();
        argument res_val{vec_ss.front()};
        argument res_ind{vec_ss.back()};
        auto in_s     = args.front().get_shape();
        auto out_s    = vec_ss.front();
        auto in_lens  = in_s.lens();
        auto axis_dim = in_lens[axis];
        std::vector<int> indices(axis_dim);

        // compute shape
        auto comp_lens = in_lens;
        comp_lens.erase(comp_lens.begin() + axis);
        shape comp_s{in_s.type(), comp_lens};

        visit_all(res_val, args.front())([&](auto out_val, auto input) {
            res_ind.visit([&](auto out_ind) {
                shape_for_each(comp_s, [&](auto idx) {
                    std::iota(indices.begin(), indices.end(), 0);
                    auto idx1 = idx;
                    auto idx2 = idx;
                    std::stable_sort(indices.begin(), indices.end(), [&](auto i1, auto i2) {
                        idx1.insert(idx1.begin() + axis, i1);
                        idx2.insert(idx2.begin() + axis, i2);
                        auto ini1 = in_s.index(idx1);
                        auto ini2 = in_s.index(idx2);
                        return (largest) ? (input[ini1] > input[ini2])
                                         : (input[ini1] < input[ini2]);
                    });

                    // copy values to output
                    auto out_idx = idx;
                    auto in_idx  = idx;
                    for(auto i : range(indices.size()))
                    {
                        in_idx.insert(in_idx.begin() + axis, indices[i]);
                        out_idx.insert(out_idx.begin() + axis, i);
                        auto outi     = out_s.index(out_idx);
                        auto ini      = in_s.index(in_idx);
                        out_val[outi] = input[ini];
                        out_ind[outi] = indices[i];
                    }
                });
            });
        });

        return argument({res_val, res_ind});
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
