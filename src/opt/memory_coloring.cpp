#include <migraphx/memory_coloring.hpp>
#include "memory_coloring_impl.hpp"

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

void memory_coloring::apply(program& p) const
{
    if(!enabled(MIGRAPH_DISABLE_MEMORY_COLORING{}))
    {
        memory_coloring_impl opt(&p, allocation_op, verify);
        opt.run();
    }
}

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
