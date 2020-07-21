#ifndef MIGRAPHX_GUARD_RTGLIB_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONVOLUTION_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/gpu/miopen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_convolution
{
    op::convolution op;
    shared<convolution_descriptor> cd;
    miopenConvFwdAlgorithm_t algo{};
    miopenHandle_t handle = nullptr;
    uint64_t solution_id = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        // TODO: Add algo
        return op::convolution::reflect(self.op, f);
    }

    std::string name() const { return "gpu::convolution"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    shape compile(context& ctx, const shape& output_shape, std::vector<shape> inputs);
    void get_solution(context& ctx, const shape& output_shape, std::vector<shape> inputs);
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
