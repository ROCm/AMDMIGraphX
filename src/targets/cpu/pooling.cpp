#include <migraphx/config.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/op/pooling.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct dnnl_pooling : dnnl_extend_op<dnnl_pooling, dnnl::pooling_forward, op::pooling>
{
    std::vector<int> arg_map(int) const { return {MIGRAPHX_DNNL_PREFIX(ARG_SRC)}; }

    dnnl::pooling_forward::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        auto algo = op.mode == op::pooling_mode::max ? dnnl::algorithm::pooling_max
                                                     : dnnl::algorithm::pooling_avg;
        auto kdims = op.kdims();
        std::vector<size_t> padding_l(op.padding.begin(), op.padding.begin() + kdims);
        std::vector<size_t> padding_r(op.padding.begin() + kdims, op.padding.end());
        return {dnnl::prop_kind::forward_inference,
                algo,
                m.at(MIGRAPHX_DNNL_PREFIX(ARG_SRC)),
                m.at(MIGRAPHX_DNNL_PREFIX(ARG_DST)),
                to_dnnl_dims(op.stride),
                to_dnnl_dims(op.lengths),
                to_dnnl_dims(padding_l),
                to_dnnl_dims(padding_r)};
    }
};

} // namespace cpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
