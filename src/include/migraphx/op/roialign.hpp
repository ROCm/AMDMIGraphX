#ifndef MIGRAPHX_GUARD_OPERATORS_ROIALIGN_HPP
#define MIGRAPHX_GUARD_OPERATORS_ROIALIGN_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct roialign
{
    std::string mode       = "avg";
    int64_t output_height  = 1;
    int64_t output_width   = 1;
    int64_t sampling_ratio = 0;
    float spatial_scale    = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"),
                    f(self.output_height, "output_height"),
                    f(self.output_width, "output_width"),
                    f(self.sampling_ratio, "sampling_ratio"),
                    f(self.spatial_scale, "spatial_scale"));
    }

    std::string name() const { return "roialign"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3).standard();
        auto lens0 = inputs.at(0).lens();
        auto lens1 = inputs.at(1).lens();
        auto type  = inputs.at(0).type();

        std::vector<std::size_t> out_lens = lens0;
        out_lens[0]                       = lens1[0];
        out_lens[2]                       = output_height;
        out_lens[3]                       = output_width;

        return {type, out_lens};
    }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
