#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>
#include <migraphx/op/add.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_add : dnnl_extend_op<dnnl_add, dnnl::binary, op::add>
{
    dnnl::binary::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        return {dnnl::algorithm::binary_add,
                m.at(DNNL_ARG_SRC_0),
                m.at(DNNL_ARG_SRC_1),
                m.at(DNNL_ARG_DST)};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
