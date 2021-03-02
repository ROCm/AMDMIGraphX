#include <migraphx/memory_coloring.hpp>
#include "memory_coloring_impl.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void memory_coloring::apply(module& p) const
{
    if(!enabled(MIGRAPHX_DISABLE_MEMORY_COLORING{}))
    {
        memory_coloring_impl opt(&p, allocation_op, 0, verify);
        opt.run();

        auto offset_start = opt.required_bytes;
        auto sub_mods     = p.get_sub_modules();
        if(!sub_mods.empty())
        {
            for(auto& smod : sub_mods)
            {
                memory_coloring_impl opt1(smod, allocation_op, offset_start, verify);
                opt1.run();
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
