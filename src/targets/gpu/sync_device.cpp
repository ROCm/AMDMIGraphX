#include <migraphx/gpu/sync_device.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void sync_device::apply(module& m) const
{
    auto last = std::prev(m.end());
    if(last->name() == "@return")
    {
        auto inputs = last->inputs();
        if(std::any_of(inputs.begin(), inputs.end(), [](auto i) {
               return (i->name() == "hip::copy_from_gpu");
           }))
        {
            auto sync_in = m.insert_instruction(last, make_op("hip::sync_stream"), inputs);
            if(not inputs.empty())
            {
                m.replace_instruction(inputs.front(), sync_in);
            }
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
