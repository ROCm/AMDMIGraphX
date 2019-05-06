#include <migraphx/horizontal_fusion.hpp>
#include "horizontal_fusion_impl.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void horizontal_fusion::apply(program& p) const
{
    if(!enabled(MIGRAPHX_DISABLE_HORIZONTAL_FUSION{}))
    {
        horizontal_fusion_impl opt(&p);
        opt.run();
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
