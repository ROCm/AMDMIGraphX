#ifndef MIGRAPHX_GUARD_OPERATORS_SCALAR_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCALAR_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scalar
{
    std::vector<std::size_t> scalar_bcast_lens;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.scalar_bcast_lens, "scalar_bcst_dims"));
    }

    std::string name() const { return "scalar"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        assert(check_shapes{inputs}.has(1).only_dims(1).size() == 1);
        auto t = inputs.at(0).type();
        std::vector<std::size_t> strides(scalar_bcast_lens.size(), 0);
        return {t, scalar_bcast_lens, strides};
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
