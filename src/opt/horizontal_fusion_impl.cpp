#include "horizontal_fusion_impl.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void horizontal_fusion_impl::run()
{
    MIGRAPHX_DEBUG(dump("---Before horizontal fusion---"));
    MIGRAPHX_DEBUG(dump_program());
}

#ifdef MIGRAPHX_DEBUG_OPT

void horizontal_fusion_impl::dump_program() { std::cout << *p_program << std::endl; }
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
