#include <migraphx/gpu/sync_device.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void sync_device::apply(program& p) const
{
    auto last = std::prev(p.end());
    if(last->name() == "@return")
    {
        auto inputs = last->inputs();
        if(std::any_of(inputs.begin(), inputs.end(), [](auto i) {
               return (i->name() == "hip::copy_from_gpu");
           }))
        {
            p.insert_instruction(last, hip_sync_device{}, inputs);
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
