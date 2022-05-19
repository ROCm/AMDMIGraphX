#include <unordered_set>
#include <migraphx/normalize_attributes.hpp>
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

void normalize_ops::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        auto inputs = ins->inputs();
        if(inputs.empty())
            continue;

        auto s                       = inputs[0]->get_shape();
        migraphx::operation tuned_op = ins->get_operator();
        if(normalize_attributes(tuned_op, s))
        {
            m.replace_instruction(ins, tuned_op, inputs);
            ins->set_normalized();
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
