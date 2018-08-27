#include <migraph/memory_coloring.hpp>
#include "memory_coloring_impl.hpp"

namespace migraph {
void memory_coloring::apply(program& p) const
{
    memory_coloring_impl opt(&p);
    opt.run();
}
} // namespace migraph
