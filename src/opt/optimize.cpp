#include <migraph/optimize.hpp>
#include "memory_coloring.hpp"

namespace migraph {
void optimize::apply(program &p) const
{
    std::cout << p << std::endl;
    memory_coloring opt(&p);
    opt.run();
    int ins_enum = p.get_size();
    if (ins_enum == 0)
        return;
    instruction_ref iter_ins = std::prev(p.end());
    instruction_ref first_ins = p.begin();
    do {
        iter_ins = std::prev(iter_ins);
        
    } while (iter_ins != first_ins);
    
}
} // namespace migraph
