#ifndef MIGRAPHX_GUARD_OPERATORS_NONZERO_HPP
#define MIGRAPHX_GUARD_OPERATORS_NONZERO_HPP

#include "migraphx/shape_for_each.hpp"
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/float_equal.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct nonzero
{
    std::string name() const { return "nonzero"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto elem_num                     = inputs[0].elements();
        auto idx_num                      = inputs[0].lens().size();
        std::vector<std::size_t> out_lens = {idx_num, elem_num};

        return {shape::int64_type, out_lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        std::vector<std::vector<std::size_t>> indices;
        auto s = args.front().get_shape();
        args.front().visit([&](auto v) {
            using type = typename decltype(v)::value_type;
            shape_for_each(s, [&](auto idx) {
                if(not float_equal(v[s.index(idx)], type{0}))
                {
                    indices.push_back(idx);
                }
            });
        });

        auto out_lens = output_shape.lens();
        out_lens[1]   = indices.size();
        shape out_s{shape::int64_type, out_lens};
        argument result{out_s};
        result.visit([&](auto output) {
            shape_for_each(out_s,
                           [&](auto idx) { output[out_s.index(idx)] = indices[idx[1]][idx[0]]; });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
