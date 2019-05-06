#include <migraphx/gpu/pack_int8_args.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/gpu/hip.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void pack_int8_args::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "gpu::quant_gemm")
            continue;

        auto inputs = ins->inputs();
        auto shape_a = inputs.at(0)->get_shape();
        if (shape_a.type() != shape::int8_type)
            continue;

        if (shape_a.transposed())
        {
            auto pack_a = p.insert_instruction(ins, hip_allocate{shape_a});
            inputs.push_back(pack_a);
            swap(inputs.at(0), inputs.back());
        }

        auto shape_b = inputs.at(1)->get_shape();
        if (!shape_b.transposed())
        {
            auto pack_b = p.insert_instruction(ins, hip_allocate{shape_b});
            inputs.push_back(pack_b);
            swap(inputs.at(1), inputs.back());
        }
        instruction::replace(ins, ins->get_operator(), ins->get_shape(), inputs);
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
