#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>
#include <migraphx/op/mul.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template struct cpu_binary<op::mul>;

#if USE_DNNL
struct dnnl_mul : dnnl_extend_op<dnnl_mul, dnnl::binary, op::mul>
{
    dnnl::binary::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {dnnl::algorithm::binary_mul,
                m.at(DNNL_ARG_SRC_0),
                m.at(DNNL_ARG_SRC_1),
                m.at(DNNL_ARG_DST)};
    }
};
#endif

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
