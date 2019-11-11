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

void preallocate_param::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "@param")
            continue;
        std::string id = any_cast<builtin::param>(ins->get_operator()).parameter;
        if(id != param)
            continue;
        argument a                                   = allocate_gpu(ins->get_shape());
        ctx->get_current_device().preallocations[id] = a;
        auto r = p.insert_instruction(ins, hip_load_memory{a.get_shape(), id});
        p.replace_instruction(ins, r);
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
