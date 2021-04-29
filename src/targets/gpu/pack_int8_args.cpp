#include <iterator>
#include <migraphx/gpu/pack_int8_args.hpp>
#include <migraphx/gpu/int8_gemm_pack.hpp>
#include <migraphx/gpu/int8_conv_pack.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/pad.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void pack_int8_args::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() == "gpu::quant_gemm")
        {
            auto&& op = any_cast<rocblas_gemm<op::quant_dot>>(ins->get_operator());
            if(not op.int8_x4_format)
            {
                return;
            }

            auto inputs = ins->inputs();
            auto lens   = inputs.at(0)->get_shape().lens();
            // gemm need the k to be multiple of 4, so need packing that dimension
            auto old_inputs = inputs;
            if((lens.back() % 4) != 0)
            {
                inputs = pad_inputs(p, ins);
            }

            bool transa = inputs[0]->get_shape().transposed();
            bool transb = inputs[1]->get_shape().transposed();
            if(!transb)
            {
                auto packed_b = p.insert_instruction(ins, hip_allocate{inputs[1]->get_shape()});
                auto output_b =
                    p.insert_instruction(ins, hip_int8_gemm_pack_a{}, {inputs[1], packed_b});
                inputs[1] = output_b;
            }

            if(transa)
            {
                auto packed_a = p.insert_instruction(ins, hip_allocate{inputs[0]->get_shape()});
                auto output_a =
                    p.insert_instruction(ins, hip_int8_gemm_pack_b{}, {inputs[0], packed_a});
                inputs[0] = output_a;
            }

            if(inputs != old_inputs)
            {
                p.replace_instruction(ins, ins->get_operator(), inputs);
            }
        }
        else if(ins->name() == "gpu::quant_convolution")
        {
            auto inputs = ins->inputs();
            auto packed_x =
                p.insert_instruction(ins, hip_allocate{pack_int8_shape(inputs[0]->get_shape())});
            auto output_x =
                p.insert_instruction(ins, miopen_int8_conv_pack{}, {inputs[0], packed_x});
            instruction::replace_argument(ins, inputs[0], output_x);

            auto packed_w =
                p.insert_instruction(ins, hip_allocate{pack_int8_shape(inputs[1]->get_shape())});
            auto output_w =
                p.insert_instruction(ins, miopen_int8_conv_pack{}, {inputs[1], packed_w});
            instruction::replace_argument(ins, inputs[1], output_w);
        }
    }
}

shape pack_int8_args::pack_int8_shape(const shape& s) const
{
    if(s.type() != shape::int8_type)
    {
        MIGRAPHX_THROW("PACK_INT8_ARGS: only process int8_type");
    }

    auto lens    = s.lens();
    auto strides = s.strides();
    lens[1]      = (lens[1] + 3) / 4 * 4;
    strides[0]   = strides[1] * lens[1];

    return {s.type(), lens, strides};
}

instruction_ref pack_int8_args::pad_ins(module& p, instruction_ref ins, int offset) const
{
    auto s                         = ins->get_shape();
    auto lens                      = s.lens();
    auto k                         = lens[lens.size() + offset];
    auto pad_k                     = (k + 3) / 4 * 4;
    auto pad_lens                  = lens;
    pad_lens[lens.size() + offset] = pad_k;
    std::vector<int64_t> pad_dims(lens.size() * 2, 0);
    auto ret_ins = ins;
    if(pad_k != k)
    {
        pad_dims[lens.size() + offset] = pad_k - k;
        shape ps{s.type(), pad_lens};
        auto ins_out = p.insert_instruction(ins, hip_allocate{ps});
        op::pad pad{pad_dims};
        ret_ins = p.insert_instruction(std::next(ins), hip_pad{pad}, ins, ins_out);
    }

    return ret_ins;
}

std::vector<instruction_ref> pack_int8_args::pad_inputs(module& p, instruction_ref ins) const
{
    std::vector<instruction_ref> ret_inputs;
    auto inputs = ins->inputs();
    auto in0    = inputs.at(0);
    auto sa     = in0->get_shape();
    bool transa = sa.transposed();
    if(transa)
    {
        auto perm  = find_permutation(sa);
        auto t_in  = in0->inputs().front();
        int offset = static_cast<int>(perm.back()) - static_cast<int>(perm.size());
        auto p_in  = pad_ins(p, t_in, offset);
        auto val   = in0->get_operator().to_value();
        assert(val.contains("dims"));
        auto dims = val.at("dims").to_vector<int64_t>();
        auto r_in = p.insert_instruction(ins, make_op("transpose", {{"dims", dims}}), p_in);
        ret_inputs.push_back(r_in);
    }
    else
    {
        ret_inputs.push_back(pad_ins(p, in0, -1));
    }

    auto in1    = inputs.at(1);
    auto sb     = in1->get_shape();
    bool transb = sb.transposed();
    if(transb)
    {
        auto perm  = find_permutation(sb);
        auto t_in  = in1->inputs().front();
        int offset = static_cast<int>(perm[perm.size() - 2]) - static_cast<int>(perm.size());
        auto p_in  = pad_ins(p, t_in, offset);
        auto val   = in1->get_operator().to_value();
        assert(val.contains("dims"));
        auto dims = val.at("dims").to_vector<int64_t>();
        auto r_in = p.insert_instruction(ins, make_op("transpose", {{"dims", dims}}), p_in);
        ret_inputs.push_back(r_in);
    }
    else
    {
        ret_inputs.push_back(pad_ins(p, in1, -2));
    }
    std::copy(inputs.begin() + 2, inputs.end(), std::back_inserter(ret_inputs));

    return ret_inputs;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
