#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTERND_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTERND_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scatternd
{
    std::string reduction = "none";

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.reduction, "reduction"));
    }

    std::string name() const { return "scatternd"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3).standard();
        return inputs.front();
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[2])([&](auto output, auto data, auto updates) {
            std::copy(data.begin(), data.end(), output.begin());
            args[1].visit([&](auto indices) {
                auto updates_shape = updates.get_shape();
                // k = index length, r = rank(data)
                // k<r => update slices, k=r => update elements
                auto k = indices.get_shape().lens().back();
                par_for(updates_shape.elements(), [&](const auto i) {
                    // updates and indices share the first dimension
                    auto offset = updates_shape.multi(i).front();
                    auto* index_start = indices.data() + (offset * k);
                    auto* index_end = index_start + k;
                    auto out_idx = output_shape.multi(i);
                    std::copy(index_start, index_end, out_idx.begin());
                    if(reduction == "add")
                        output[output_shape.index(out_idx)] += updates[i];
                    else if (reduction == "mul")
                        output[output_shape.index(out_idx)] *= updates[i];
                    else
                        output[output_shape.index(out_idx)] = updates[i];
                });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
