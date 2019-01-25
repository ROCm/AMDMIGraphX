#include <migraphx/gpu/eliminate_set_stream.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/pass_config.hpp>
#include <migraphx/gpu/event.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void eliminate_set_stream::apply(program& p) const
{
    int last_stream = -1;
    std::vector<instruction_ref> instrs;
    for(auto ins : iterator_for(p))
    {
        if (ins->name() != "gpu::set_stream")
            continue;
        int stream = any_cast<gpu::set_stream>(ins->get_operator()).stream;
        if (stream != last_stream) {
            last_stream = stream;
            continue;
        }
        instrs.push_back(ins);
    }

    for (auto&& ins : instrs)
    {
        p.remove_instruction(ins);
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
