#include <migraph/gpu/eliminate_allocation.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/ranges.hpp>
#include <migraph/stringutils.hpp>

namespace migraph {
namespace gpu {

void eliminate_allocation::apply(program& p) const
{
    std::size_t n = 0;
    std::vector<std::pair<instruction_ref, std::size_t>> allocs;
    for(auto ins : iterator_for(p))
    {
        if(ins->op.name() != "hip::allocate")
            continue;
        allocs.emplace_back(ins, n);
        std::size_t size = ins->get_shape().bytes();
        n += size + (size % 4);
    }
    auto mem = p.add_parameter("memory", shape{shape::int8_type, {n}});
    for(auto&& pp : allocs)
    {
        auto ins = pp.first;
        auto s = ins->get_shape();
        auto offset = pp.second;
        p.replace_instruction(ins, hip_load{s, offset}, mem);
    }
}
} // namespace gpu
} // namespace migraph
