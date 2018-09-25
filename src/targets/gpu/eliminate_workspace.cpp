#include <migraph/gpu/eliminate_workspace.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/ranges.hpp>
#include <migraph/stringutils.hpp>

namespace migraph {
namespace gpu {

void eliminate_workspace::apply(program& p) const
{
    std::size_t n = 0;
    std::vector<instruction_ref> allocs;
    for(auto ins : iterator_for(p))
    {
        if(ins->outputs().size() != 1)
            continue;
        if(ins->name() != "hip::allocate")
            continue;
        auto&& a = any_cast<hip_allocate>(ins->get_operator());
        if(a.tag == "workspace")
        {
            n = std::max(n, ins->get_shape().bytes());
            allocs.push_back(ins);
        }
    }
    auto ws = p.add_parameter("workspace", shape{shape::int8_type, {n}});
    for(auto&& a : allocs)
    {
        p.replace_instruction(a, ws);
        p.remove_instruction(a);
    }
}
} // namespace gpu
} // namespace migraph
