#include <migraphx/onnx/conv.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

void recalc_conv_attributes(value& v, size_t kdims)
{
    if(not(v["padding"].size() == kdims or v["padding"].size() == kdims * 2))
    {
        v["padding"].resize(kdims);
        std::fill_n(v["padding"].begin(), kdims, 0);
    }
    if(v["stride"].size() != kdims)
    {
        v["stride"].resize(kdims);
        std::fill_n(v["stride"].begin(), kdims, 1);
    }
    if(v["dilation"].size() != kdims)
    {
        v["dilation"].resize(kdims);
        std::fill_n(v["dilation"].begin(), kdims, 1);
    }
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
