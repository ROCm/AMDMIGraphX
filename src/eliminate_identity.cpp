#include <migraphx/eliminate_identity.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Range, class Iterator>
std::ptrdiff_t bidistance(const Range& r, Iterator start, Iterator last)
{
    auto start_forward   = start;
    auto start_backwards = start;
    std::size_t n        = 0;
    while(start_forward != last and start_backwards != last)
    {
        n++;
        if(start_forward != r.end())
            start_forward++;
        if(start_backwards != r.begin())
            start_backwards--;
    }
    if(start_forward == last)
        return n;
    else
        return -n;
}

void eliminate_identity::apply(program& p) const
{
    auto last = std::prev(p.end());
    for(auto ins : iterator_for(p))
    {
        // Skip the first instruction, since we always process the previous
        // instruction
        if(ins == p.begin())
            continue;
        const auto i = std::prev(ins);

        if(i->name() == "identity")
        {
            p.replace_instruction(i, i->inputs().front());
            p.move_instruction(i, p.end());
        }
        if(ins == last)
        {
            if(ins->name() == "identity")
            {
                const instruction_ref& identity_input = i->inputs().front();
                if(identity_input->outputs().size() == 1)
                {
                    p.move_instruction(identity_input, i);
                    // since this is the last instruction, removing it only
                    // requires changing "last" and calling remove below
                    last = std::prev(last);
                }
            }
            break;
        }
    }
    p.remove_instructions(std::next(last), p.end());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
