#include "migraphx/instruction_ref.hpp"
#include <migraphx/gpu/pack_int8_args.hpp>
#include <migraphx/gpu/int8_gemm_pack.hpp>
#include <migraphx/gpu/int8_conv_pack.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
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
            if(not op.int8X4_format)
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

            std::cout << "loc20" << std::endl;
            bool transa = inputs[0]->get_shape().transposed();
            bool transb = inputs[1]->get_shape().transposed();

            if(!transb)
            {
                std::cout << "loc21" << std::endl;
                auto packed_b = p.insert_instruction(ins, hip_allocate{inputs[1]->get_shape()});
                std::cout << "loc22" << std::endl;
                auto output_b =
                    p.insert_instruction(ins, hip_int8_gemm_pack_a{}, {inputs[1], packed_b});
                inputs[1] = output_b;
                // std::cout << "loc23" << std::endl;
                // instruction::replace_argument(ins, old_inputs[1], output_b);
                // std::cout << "loc24" << std::endl;
            }

            if(transa)
            {
                std::cout << "loc25" << std::endl;
                auto packed_a = p.insert_instruction(ins, hip_allocate{inputs[0]->get_shape()});
                std::cout << "loc26" << std::endl;
                auto output_a =
                    p.insert_instruction(ins, hip_int8_gemm_pack_b{}, {inputs[0], packed_a});
                inputs[0] = output_a;
                // std::cout << "loc27" << std::endl;
                // instruction::replace_argument(ins, old_inputs[0], output_a);
                // std::cout << "loc28" << std::endl;
            }

            std::cout << "p1 = " << std::endl;
            std::cout << p << std::endl;
            if(inputs != old_inputs)
            {
                std::cout << "loc27, input_size = " << inputs.size() << std::endl;
                p.replace_instruction(ins, ins->get_operator(), inputs);
                std::cout << "loc27" << std::endl;
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

std::vector<instruction_ref> pack_int8_args::pad_inputs(module& p, instruction_ref ins) const
{
    std::vector<instruction_ref> ret_inputs;
    auto inputs = ins->inputs();
    auto sa     = inputs.at(0)->get_shape();
    auto alens  = sa.lens();
    bool transa = sa.transposed();
    std::cout << "loc1" << std::endl;
    if(transa)
    {
        std::cout << "loc2" << std::endl;
        auto perm  = find_permutation(sa);
        auto r_in  = inputs.at(0);
        auto t_in  = r_in->inputs().front();
        auto tlens = t_in->get_shape().lens();
        auto k     = tlens[perm.back()];
        auto pad_k = (k + 3) / 4 * 4;
        std::vector<int64_t> pad_dims(tlens.size() * 2, 0);
        pad_dims[perm.back()] = pad_k - k;
        auto val              = inputs.at(0)->get_operator().to_value();
        assert(val.contains("dims"));
        if(pad_k != k)
        {
            auto perm1 = val.at("dims").to_vector<int64_t>();
            t_in       = p.insert_instruction(ins, make_op("pad", {{"pads", pad_dims}}), t_in);
            r_in       = p.insert_instruction(ins, make_op("transpose", {{"dims", perm1}}), t_in);
        }
        ret_inputs.push_back(r_in);
        std::cout << "loc3" << std::endl;
    }
    else
    {
        std::cout << "loc4" << std::endl;
        auto k     = alens.back();
        auto pad_k = (k + 3) / 4 * 4;
        std::vector<int64_t> pad_dims(alens.size() * 2, 0);
        pad_dims[alens.size() - 1] = pad_k - k;
        auto inp_0                 = inputs.at(0);
        if(pad_k != k)
        {
            std::cout << "loc5" << std::endl;
            inp_0 = p.insert_instruction(ins, make_op("pad", {{"pads", pad_dims}}), inp_0);
        }
        ret_inputs.push_back(inp_0);
        std::cout << "loc6" << std::endl;
    }

    auto sb     = inputs.at(1)->get_shape();
    auto blens  = sb.lens();
    bool transb = sb.transposed();
    if(transb)
    {
        auto perm  = find_permutation(sb);
        auto r_in  = inputs.at(1);
        auto t_in  = r_in->inputs().front();
        auto tlens = t_in->get_shape().lens();
        auto k     = tlens[perm.size() - 1];
        auto pad_k = (k + 3) / 4 * 4;
        std::vector<int64_t> pad_dims(tlens.size() * 2, 0);
        pad_dims[perm.size() - 2] = pad_k - k;
        auto val                  = inputs.at(0)->get_operator().to_value();
        std::cout << "loc7" << std::endl;
        assert(val.contains("dims"));
        if(pad_k != k)
        {
            std::cout << "loc8" << std::endl;
            auto perm1 = val.at("dims").to_vector<int64_t>();
            t_in       = p.insert_instruction(ins, make_op("pad", {{"pads", pad_dims}}), t_in);
            r_in       = p.insert_instruction(ins, make_op("transpose", {{"dims", perm1}}), t_in);
            std::cout << "loc9" << std::endl;
        }
        ret_inputs.push_back(r_in);
        std::cout << "loc10" << std::endl;
    }
    else
    {
        std::cout << "loc11" << std::endl;
        auto k     = blens[blens.size() - 2];
        auto pad_k = (k + 3) / 4 * 4;
        std::vector<int64_t> pad_dims(blens.size() * 2, 0);
        pad_dims[blens.size() - 2] = pad_k - k;
        auto inp_1                 = inputs.at(1);
        if(pad_k != k)
        {
            std::cout << "loc12" << std::endl;
            inp_1 = p.insert_instruction(ins, make_op("pad", {{"pads", pad_dims}}), inp_1);
        }
        ret_inputs.push_back(inp_1);
        std::cout << "loc13" << std::endl;
    }

    std::cout << "p = " << std::endl;
    std::cout << p << std::endl;

    return ret_inputs;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
