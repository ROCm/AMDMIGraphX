#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/load.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/pass_config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_allocation::apply(program& p) const
{
    assert(alignment > 0);

    std::size_t n = 0;
    std::vector<std::pair<instruction_ref, std::size_t>> allocs;
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != allocation_op)
            continue;
        allocs.emplace_back(ins, n);
        std::size_t size    = ins->get_shape().bytes();
        std::size_t padding = (alignment - (size % alignment)) % alignment;
        n += size + padding;
    }
    if(n > 0)
    {
        auto mem = p.add_parameter("memory", shape{shape::int8_type, {n}});
        for(auto&& pp : allocs)
        {
            auto ins    = pp.first;
            auto s      = ins->get_shape();
            auto offset = pp.second;
            p.replace_instruction(ins, op::load{s, offset}, mem);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
