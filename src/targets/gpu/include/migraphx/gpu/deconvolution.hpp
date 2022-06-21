#ifndef MIGRAPHX_GUARD_RTGLIB_DECONVOLUTION_HPP
#define MIGRAPHX_GUARD_RTGLIB_DECONVOLUTION_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/deconvolution.hpp>
#include <migraphx/gpu/miopen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_deconvolution
{
    op::deconvolution op;
    shared<convolution_descriptor> cd;
    miopenConvFwdAlgorithm_t algo{};
    miopenHandle_t handle = nullptr;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        // TODO: Add algo
        return op::convolution::reflect(self.op, f);
    }

    std::string name() const { return "gpu::deconv"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    shape compile(context& ctx, const shape& output_shape, std::vector<shape> inputs);
    void finalize(context& ctx, const shape& output_shape, std::vector<shape> inputs);
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
