#include <migraphx/promote_precision.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static std::unordered_set<instruction_ref> find_instruction_to_upgrade(module& m, shape::type_t t)
{
    std::unordered_set<instruction_ref> result;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convert")
            continue;
        if(ins->inputs().front()->get_shape().type() != t)
            continue;
    }
    return result;
}

void promote_precision::apply(module_pass_manager& mpm) const {}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
