#ifndef MIGRAPHX_GUARD_OPERATORS_GATHER_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHER_HPP

#include <algorithm>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/heap.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct topk
{
    int64_t k    = 1;
    int64_t axis = 0;
    bool largest = true;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.k, "k"), f(self.axis, "axis"), f(self.largest, "largest"));
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
        auto in_s      = args.front().get_shape();
        auto out_s     = vec_ss.front();
        auto comp_lens = in_s.lens();
        auto axis_dim  = comp_lens[axis];

        // compute shape
        comp_lens[axis] = 1;
        shape comp_s{in_s.type(), comp_lens};
        visit_all(res_val, args.front())([&](auto out_val, auto input) {
            auto* out_ind = res_ind.cast<int64_t>();
            par_for(comp_s.elements(), [&](auto i) {
                auto idx = comp_s.multi(i);
                std::vector<std::size_t> indices(k);
                std::iota(indices.begin(), indices.end(), 0);

                auto comp = [&](auto i1, auto i2) {
                    auto idx1  = idx;
                    auto idx2  = idx;
                    idx1[axis] = i1;
                    idx2[axis] = i2;
                    return this->largest
                               ? std::greater<>{}(input[in_s.index(idx1)], input[in_s.index(idx2)])
                               : std::less<>{}(input[in_s.index(idx1)], input[in_s.index(idx2)]);
                };

                auto hp = make_heap(indices, comp);
                for(std::size_t ii = indices.size(); ii < axis_dim; ++ii)
                {
                    hp.update(ii);
                }
                hp.sort();

                auto sorted_indices = hp.get();
                auto out_idx        = idx;
                auto in_idx         = idx;
                for(auto j : range(sorted_indices.size()))
                {
                    out_idx[axis]                 = j;
                    in_idx[axis]                  = sorted_indices[j];
                    out_val[out_s.index(out_idx)] = input[in_s.index(in_idx)];
                    out_ind[out_s.index(out_idx)] = sorted_indices[j];
                }
            });
        });

        return argument({res_val, res_ind});
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
