#include <migraphx/memory_coloring.hpp>
#include "memory_coloring_impl.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void memory_coloring::apply(program& p) const
{
    if(!enabled(MIGRAPHX_DISABLE_MEMORY_COLORING{}))
    {
        memory_coloring_impl opt(&p, allocation_op, verify, num_of_streams, f_concur);
        opt.run();
    }
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
