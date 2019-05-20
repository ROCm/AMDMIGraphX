#include <migraphx/auto_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/contiguous.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void auto_contiguous::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        shape s = ins->get_shape();
        if(not s.standard() and s.elements() != 0)
        {
            auto c = p.insert_instruction(std::next(ins), op::contiguous{}, ins);
            p.replace_instruction(ins, c);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
