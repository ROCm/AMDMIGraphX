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
        auto idx_num                      = inputs[0].lens().size();
        std::vector<std::size_t> out_lens = {idx_num, elem_num};

        return {shape::int64_type, out_lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto elem_num = args.front().get_shape().elements();
        std::vector<int64_t> vec_idx(elem_num);
        args.front().visit([&](auto data) {
            par_for(elem_num, [&](auto i) { vec_idx[i] = (float_equal(data[i], 0)) ? 0 : 1; });
        });

        std::partial_sum(vec_idx.begin(), vec_idx.end(), vec_idx.begin());

        auto s        = args.front().get_shape();
        auto out_lens = output_shape.lens();
        argument result{output_shape};
        result.visit([&](auto output) {
            std::fill(output.begin(), output.end(), 0);
            par_for(elem_num, [&](auto i) {
                auto nz  = static_cast<std::size_t>(vec_idx[i] - 1);
                auto idx = s.multi(nz);
                for(std::size_t j = 0; j < idx.size(); ++j)
                {
                    output[output_shape.index({j, nz})] = idx[j];
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
