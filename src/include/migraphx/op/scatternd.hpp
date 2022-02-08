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
                auto indices_shape = indices.get_shape();
                auto k             = indices_shape.lens().back();
                auto q             = indices_shape.lens().size();
                auto r             = output_shape.lens().size();
                par_for(updates_shape.elements(), [&](const auto i) {
                    auto updates_idx = updates_shape.multi(i);
                    std::vector<std::size_t> indices_idx(q, 0);
                    std::copy(
                        updates_idx.begin(), updates_idx.begin() + q - 1, indices_idx.begin());
                    auto* index_start = indices.data() +
                                        indices_shape.index(indices_idx.begin(), indices_idx.end());
                    auto* index_end = index_start + k;

                    std::vector<std::size_t> out_idx(r, 0);
                    std::copy(index_start, index_end, out_idx.begin());
                    std::copy(updates_idx.begin() + q - 1, updates_idx.end(), out_idx.begin() + k);

                    if(reduction == "add")
                        output[output_shape.index(out_idx)] += updates[i];
                    else if(reduction == "mul")
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
