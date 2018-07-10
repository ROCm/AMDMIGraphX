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
        // Skip the last instruction
        if(std::prev(ins) == last)
            break;
        fix([&](auto self, auto leaf) {
            assert(p.has_instruction(leaf));
            if(leaf->output.empty())
            {
                auto args = leaf->arguments;
                leaf->clear_arguments();
                p.move_instruction(leaf, p.end());
                for(auto arg : args)
                    self(arg);
            }
        })(std::prev(ins));
    }
    p.remove_instructions(std::next(last), p.end());
}

} // namespace migraph
