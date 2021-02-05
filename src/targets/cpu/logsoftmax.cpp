#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/op/logsoftmax.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_logsoftmax : dnnl_extend_op<dnnl_logsoftmax, dnnl::logsoftmax_forward, op::logsoftmax>
{
    dnnl::logsoftmax_forward::desc
    get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        int axis = this->op.axis;
        return {dnnl::prop_kind::forward_inference, m.at(DNNL_ARG_SRC_0), axis};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
