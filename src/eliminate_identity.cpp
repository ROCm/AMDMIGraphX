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

void eliminate_identity::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        std::vector<instruction_ref> new_ins_inputs = ins->inputs();
        // check each input arg for identity ops,
        // replace with the input of the respective identity
        for(instruction_ref& input : new_ins_inputs)
        {
            if(input->name() == "identity")
            {
                input = input->inputs().at(0);
            }
        }
        if(new_ins_inputs != ins->inputs())
            p.replace_instruction(ins, ins->get_operator(), new_ins_inputs);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
