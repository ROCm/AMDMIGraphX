#ifndef MIGRAPHX_GUARD_OPERATORS_NONZERO_HPP
#define MIGRAPHX_GUARD_OPERATORS_NONZERO_HPP

#include <migraphx/shape_for_each.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/par_for.hpp>
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
        auto dim_num                      = inputs[0].lens().size();
        std::vector<std::size_t> out_lens = {dim_num, elem_num};

        return {shape::int64_type, out_lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        std::vector<std::vector<std::size_t>> vec_idx;
        auto s = args.front().get_shape();
        args.front().visit([&](auto v) {
            shape_for_each(s, [&](auto idx) {
                if(not float_equal(v[s.index(idx)], 0))
                {
                    vec_idx.push_back(idx);
                }
            });
        });

        argument result{output_shape};
        result.visit([&](auto output) {
            std::fill(output.begin(), output.end(), 0);
            par_for(vec_idx.size(), [&](auto i) {
                for(std::size_t j = 0; j < vec_idx.front().size(); ++j)
                {
                    output[output_shape.index({j, i})] = vec_idx[i][j];
                }
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
