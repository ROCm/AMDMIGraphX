#ifndef MIGRAPHX_GUARD_OPERATORS_BATCH_NORM_HPP
#define MIGRAPHX_GUARD_OPERATORS_BATCH_NORM_HPP

#include <array>
#include <migraphx/op/common.hpp>
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

struct batch_norm_inference
{
    float epsilon  = 1.0e-6f;
    float momentum = 0.9f;

    std::string name() const { return "batch_norm_inference"; }

    enum bn_infer_mode_t
    {
        per_activation,
        spatial,
    };

    bn_infer_mode_t bn_mode = spatial;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.epsilon, "epsilon"), f(self.momentum, "momentum"), f(self.bn_mode, "bn_mode"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        check_shapes{inputs.data(), inputs.data() + 1, *this}.same_ndims();
        check_shapes{inputs.data() + 1, inputs.data() + inputs.size(), *this}.same_shape();
        return inputs.front();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
