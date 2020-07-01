#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_GEMM_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_GEMM_HPP

#include <migraphx/shape.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/gemm_impl.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

template <class Op>
struct rocblas_gemm
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const
    {
        if(contains(op.name(), "quant_"))
        {
            return "gpu::quant_gemm";
        }
        return "gpu::gemm";
    }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        std::vector<shape> in_shapes(inputs);
        in_shapes.pop_back();
        check_shapes{in_shapes}.not_broadcasted();
        batch_not_transposed(inputs[0].strides());
        batch_not_transposed(inputs[1].strides());

        return op.compute_shape(in_shapes);
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        gemm(ctx, output_shape, args, op.alpha, op.beta);
        return args.back();
    }

    void batch_not_transposed(const std::vector<std::size_t>& strides) const
    {
        if(strides.size() <= 2)
            return;
        auto dim_0       = strides.size() - 2;
        auto matrix_size = std::max(strides[dim_0], strides[dim_0 + 1]);
        std::vector<std::size_t> batch(strides.begin(), strides.begin() + dim_0);
        if(std::all_of(batch.begin(), batch.end(), [&](auto i) { return (i < matrix_size); }))
        {
            MIGRAPHX_THROW("GPU_GEMM: matrix size and batch size {" + to_string_range(strides) +
                           "} are transposed!");
        }

        if(std::adjacent_find(batch.begin(), batch.end(), [&](auto i, auto j) {
               return (i < j or i < matrix_size or j < matrix_size);
           }) != batch.end())
        {
            MIGRAPHX_THROW("GPU_GEMM: batch size {" + to_string_range(strides) +
                           "} is transposed!");
        }
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
