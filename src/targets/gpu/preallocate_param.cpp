#include <migraphx/gpu/preallocate_param.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void preallocate_param::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "@param")
            continue;
        if(param != any_cast<builtin::param>(ins->get_operator()).parameter)
            continue;
        std::string id = p.name() + ":" + param;
        auto r         = p.insert_instruction(ins, hip_allocate_memory{ins->get_shape(), id});
        p.replace_instruction(ins, r);
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
