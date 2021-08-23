#include <migraphx/preallocate_param.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void preallocate_param::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "@param")
            continue;
        if(param != any_cast<builtin::param>(ins->get_operator()).parameter)
            continue;
        std::string id = m.name() + ":" + param;
        auto r         = m.insert_instruction(ins, model.preallocate(ins->get_shape(), id));
        m.replace_instruction(ins, r);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
