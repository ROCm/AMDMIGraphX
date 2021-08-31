#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/op/lrn.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_lrn : dnnl_extend_op<dnnl_lrn, dnnl::lrn_forward, op::lrn>
{
    dnnl::lrn_forward::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {dnnl::prop_kind::forward_inference,
                dnnl::algorithm::lrn_across_channels,
                m.at(MIGRAPHX_DNNL_PREFIX(ARG_SRC_0)),
                this->op.size,
                this->op.alpha,
                this->op.beta,
                this->op.bias};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
