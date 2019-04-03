#ifndef MIGRAPHX_GUARD_OPERATORS_PAD_HPP
#define MIGRAPHX_GUARD_OPERATORS_PAD_HPP

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

struct pad
{
    std::vector<int64_t> pads;
    float value = 0.0f;
    enum pad_op_mode_t
    {
        constant_pad,
        reflect_pad,
        edge_pad
    };
    pad_op_mode_t mode = constant_pad;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"), f(self.pads, "pads"), f(self.value, "value"));
    }

    std::string name() const { return "pad"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto&& idims = inputs.front().lens();
        std::vector<std::size_t> rdims(idims.begin(), idims.end());
        std::size_t num_dims = rdims.size();

        for(std::size_t i = 0; i < num_dims; i++)
        {
            rdims[i] += pads[i] + pads[i + num_dims];
        }

        shape s{inputs.front().type(), rdims};
        return s;
    }

    bool symmetric() const
    {
        std::size_t num_dims = pads.size() / 2;
        return std::equal(
            pads.begin(), pads.begin() + num_dims, pads.begin() + num_dims, pads.end());
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
