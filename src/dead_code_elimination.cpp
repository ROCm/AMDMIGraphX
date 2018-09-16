#include <migraph/dead_code_elimination.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/functional.hpp>

namespace migraph {

void dead_code_elimination::apply(program& p) const
{
    auto last = std::prev(p.end());
    for(auto ins : iterator_for(p))
    {
        // Skip the first instruction, since we always process the previous
        // instruction
        if(ins == p.begin())
            continue;
        const auto i = std::prev(ins);
        // Skip instruction with empty shape as output unless its a builtin
        if(i->get_shape().elements() == 0 and not(i->name().front() == '@'))
            continue;
        // Skip the last instruction
        if(i == last)
            break;
        fix([&](auto self, auto leaf) {
            assert(p.has_instruction(leaf));
            if(leaf->outputs().empty())
            {
                auto args = leaf->inputs();
                leaf->clear_arguments();
                p.move_instruction(leaf, p.end());
                for(auto arg : args)
                    self(arg);
            }
        })(i);
    }
    p.remove_instructions(std::next(last), p.end());
}

} // namespace migraph
