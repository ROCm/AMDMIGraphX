#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>
#include <migraphx/op/relu.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template struct cpu_unary<op::relu>;

#if USE_DNNL
struct dnnl_relu : dnnl_extend_op<dnnl_relu, dnnl::eltwise_forward, op::relu>
{
    dnnl::eltwise_forward::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                           dnnl::algorithm::eltwise_relu,
                                           m.at(DNNL_ARG_SRC_0));
    }
};
#endif

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
