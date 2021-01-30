#include <migraphx/config.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/cpu/migemm.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/quant_dot.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_gemm : dnnl_extend_op<dnnl_gemm, dnnl::matmul, op::dot>
{
    std::vector<int> arg_map(int) const { return {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS}; }

    // Batch must be a single dimension
    shape adjust_shape(shape x, int) const
    {
        auto s     = base_adjust_shape(x);
        auto ndims = s.lens().size();
        if(ndims > 3)
        {
            if(not std::is_sorted(
                   s.strides().begin(), s.strides().begin() + (ndims - 2), std::greater<>{}))
                MIGRAPHX_THROW("Batch transposed");
            std::size_t batch = std::accumulate(
                s.lens().begin(), s.lens().begin() + (ndims - 2), 1, std::multiplies<>{});
            shape s3d{s.type(),
                      {batch, s.lens()[ndims - 2], s.lens()[ndims - 1]},
                      {s.lens()[ndims - 2] * s.lens()[ndims - 1],
                       s.strides()[ndims - 2],
                       s.strides()[ndims - 1]}};
            return s3d;
        }
        else
        {
            return s;
        }
    }

    dnnl::matmul::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {m.at(DNNL_ARG_SRC), m.at(DNNL_ARG_WEIGHTS), m.at(DNNL_ARG_DST)};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
