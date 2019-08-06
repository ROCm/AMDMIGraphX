#include <migraphx/gpu/pack_int8_args.hpp>
#include <migraphx/gpu/int8_gemm_pack.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void pack_int8_args::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() == "gpu::quant_gemm")
        {
            auto inputs = ins->inputs();
            bool transa = inputs[0]->get_shape().transposed();
            bool transb = inputs[1]->get_shape().transposed();

            if(!transb)
            {
                auto packed_b = p.insert_instruction(ins, hip_allocate{inputs[1]->get_shape()});
                auto output_b =
                    p.insert_instruction(ins, hip_int8_gemm_pack_a{}, {inputs[1], packed_b});
                instruction::replace_argument(ins, inputs[1], output_b);
            }

            if(transa)
            {
                auto packed_a = p.insert_instruction(ins, hip_allocate{inputs[0]->get_shape()});
                auto output_a =
                    p.insert_instruction(ins, hip_int8_gemm_pack_b{}, {inputs[0], packed_a});
                instruction::replace_argument(ins, inputs[0], output_a);
            }
        }
        else if(ins->name() == "gpu::quant_convolution")
        {
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
