#include <migraph/memory_coloring.hpp>
#include "memory_coloring_impl.hpp"

namespace migraph {
inline namespace version_1 {

void memory_coloring::apply(program& p) const
{
    if(!enabled(MIGRAPH_DISABLE_MEMORY_COLORING{}))
    {
        memory_coloring_impl opt(&p, allocation_op);
        opt.run();
    }
}

} // namespace version_1
} // namespace migraph
