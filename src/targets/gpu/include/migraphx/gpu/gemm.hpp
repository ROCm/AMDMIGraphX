#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_GEMM_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_GEMM_HPP

#include <migraphx/errors.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/value.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/reflect.hpp>
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
    float alpha         = 1;
    float beta          = 0;
    bool int8_x4_format = true;
    bool compute_fp32   = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack_join(migraphx::reflect(self.op, f),
                         pack(f(self.alpha, "alpha"),
                              f(self.beta, "beta"),
                              f(self.int8_x4_format, "int8_x4_format")));
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
        check_shapes{in_shapes, *this}.not_broadcasted();
        batch_not_transposed(inputs[0].strides());
        batch_not_transposed(inputs[1].strides());
        // if gemm and add are fused
        if(not float_equal(beta, 0))
        {
            auto cmat_shape = in_shapes.back();
            in_shapes.pop_back();
            auto op_out_shape = op.compute_shape(in_shapes);
            if(cmat_shape.lens() != op_out_shape.lens())
            {
                MIGRAPHX_THROW(this->name() + " : dimension mismatch, operand C: {" +
                               to_string_range(cmat_shape.lens()) +
                               "}, cannot add to operand A * B: {" +
                               to_string_range(op_out_shape.lens()) + "}");
            }
            if(cmat_shape.type() != op_out_shape.type())
            {
                MIGRAPHX_THROW(this->name() + " : operand C type mismatch, operand C is of type: " +
                               to_string(cmat_shape.type()) +
                               ", it must be: " + to_string(op_out_shape.type()));
            }
        }

        return op.compute_shape(in_shapes);
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        if(this->name() == "gpu::gemm")
        {
            gemm(ctx, output_shape, args, alpha, beta, int8_x4_format, compute_fp32);
        }
        else
        {
            gemm(ctx,
                 output_shape,
                 args,
                 int32_t(alpha),
                 int32_t(beta),
                 int8_x4_format,
                 compute_fp32);
        }
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
