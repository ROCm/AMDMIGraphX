#include <migraphx/quantize_ins.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void quantize_ins(program& prog, const std::vector<std::string>& ins_names)
{
    for(auto ins : iterator_for(prog))
    {
        auto name_it = std::find(ins_name.begin(), ins_name.end(), ins->name());
        if(name_it == ins_name.end())
        {
            continue;
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
