#include <migraphx/config.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/quant_dot.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_gemm : dnnl_extend_op<dnnl_gemm, dnnl::matmul, op::dot>
{
    std::vector<int> arg_map(int) const { return {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS}; }

    void required(const check_shapes& cs) const { cs.not_broadcasted(); }

    dnnl::matmul::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {m.at(DNNL_ARG_SRC), m.at(DNNL_ARG_WEIGHTS), m.at(DNNL_ARG_DST)};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
