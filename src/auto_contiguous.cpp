#include <migraphx/auto_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

void auto_contiguous::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        shape s = ins->get_shape();
        if(not s.standard())
        {
            auto c = p.insert_instruction(std::next(ins), op::contiguous{}, ins);
            p.replace_instruction(ins, c);
        }
    }
}

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
