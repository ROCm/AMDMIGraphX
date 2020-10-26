#include <unordered_set>
#include <migraphx/normalize_axes.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/auto_any_cast.hpp>
#include <migraphx/value.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void normalize_ops::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        auto inputs = ins->inputs();
        if(inputs.empty())
            continue;

        auto lens     = inputs[0]->get_shape().lens();
        migraphx::operation tuned_op = ins->get_operator();
        if (normalize_axes(tuned_op, lens))
        {
            p.replace_instruction(ins, tuned_op, inputs);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
