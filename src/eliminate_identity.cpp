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
        if(ins->name() == "identity")
            p.replace_instruction(ins, ins->inputs().front());
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
