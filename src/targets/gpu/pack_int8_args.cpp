/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <iterator>
#include <migraphx/gpu/pack_int8_args.hpp>
#include <migraphx/gpu/int8_gemm_pack.hpp>
#include <migraphx/gpu/int8_conv_pack.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static instruction_ref pad_ins(module& m, instruction_ref ins, int offset)
{
    auto s                         = ins->get_shape();
    auto lens                      = s.lens();
    auto k                         = lens[lens.size() + offset];
    auto pad_k                     = (k + 3) / 4 * 4;
    auto pad_lens                  = lens;
    pad_lens[lens.size() + offset] = pad_k;
    auto ret_ins                   = ins;
    if(pad_k != k)
    {
        std::vector<int64_t> pad_dims(lens.size() * 2, 0);
        pad_dims[lens.size() + offset] = pad_k - k;
        shape ps{s.type(), pad_lens};
        auto ins_out =
            m.insert_instruction(ins, make_op("hip::allocate", {{"shape", to_value(ps)}}));
        auto pad = make_op("pad", {{"pads", pad_dims}});
        ret_ins =
            m.insert_instruction(std::next(ins), make_op("gpu::pad", pad.to_value()), ins, ins_out);
    }

    return ret_ins;
}

static std::vector<instruction_ref> pad_inputs(module& m, instruction_ref ins)
{
    std::vector<instruction_ref> ret_inputs;
    auto inputs = ins->inputs();
    auto in0    = inputs.at(0);
    auto sa     = in0->get_shape();
    bool transa = sa.transposed();
    if(transa)
    {
        auto perm = find_permutation(sa);
        auto val  = in0->get_operator().to_value();
        if(val.contains("dims"))
        {
            int offset = static_cast<int>(perm.back()) - static_cast<int>(perm.size());
            auto t_in  = in0->inputs().front();
            auto p_in  = pad_ins(m, t_in, offset);
            auto dims  = val.at("dims").to_vector<int64_t>();
            auto r_in =
                m.insert_instruction(ins, make_op("transpose", {{"permutation", dims}}), p_in);
            ret_inputs.push_back(r_in);
        }
        else
        {
            shape cs{in0->get_shape().type(), in0->get_shape().lens()};
            auto con_out =
                m.insert_instruction(ins, make_op("hip::allocate", {{"shape", to_value(cs)}}));
            auto cin0 = m.insert_instruction(ins, make_op("gpu::contiguous"), in0, con_out);
            ret_inputs.push_back(pad_ins(m, cin0, -1));
        }
    }
    else
    {
        ret_inputs.push_back(pad_ins(m, in0, -1));
    }

    auto in1    = inputs.at(1);
    auto sb     = in1->get_shape();
    bool transb = sb.transposed();
    if(transb)
    {
        auto perm = find_permutation(sb);
        auto val  = in1->get_operator().to_value();
        if(val.contains("dims"))
        {
            int offset = static_cast<int>(perm[perm.size() - 2]) - static_cast<int>(perm.size());
            auto t_in  = in1->inputs().front();
            auto p_in  = pad_ins(m, t_in, offset);
            auto dims  = val.at("dims").to_vector<int64_t>();
            auto r_in =
                m.insert_instruction(ins, make_op("transpose", {{"permutation", dims}}), p_in);
            ret_inputs.push_back(r_in);
        }
        else
        {
            shape cs{in1->get_shape().type(), in1->get_shape().lens()};
            auto con_out =
                m.insert_instruction(ins, make_op("hip::allocate", {{"shape", to_value(cs)}}));
            auto cin1 = m.insert_instruction(ins, make_op("gpu::contiguous"), in1, con_out);
            ret_inputs.push_back(pad_ins(m, cin1, -2));
        }
    }
    else
    {
        ret_inputs.push_back(pad_ins(m, in1, -2));
    }
    std::copy(inputs.begin() + 2, inputs.end(), std::back_inserter(ret_inputs));

    return ret_inputs;
}

void pack_int8_args::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "gpu::quant_gemm")
        {
            auto val = ins->get_operator().to_value();
            assert(val.contains("int8_x4_format"));
            if(not val.at("int8_x4_format").to<bool>())
            {
                continue;
            }
            auto inputs = ins->inputs();
            auto lens   = inputs.at(0)->get_shape().lens();
            // gemm need the k to be multiple of 4, so need packing that dimension
            auto old_inputs = inputs;
            if((lens.back() % 4) != 0)
            {
                inputs = pad_inputs(m, ins);
            }

            bool transa = inputs[0]->get_shape().transposed();
            bool transb = inputs[1]->get_shape().transposed();
            if(not transb)
            {
                auto packed_b = m.insert_instruction(
                    ins, make_op("hip::allocate", {{"shape", to_value(inputs[1]->get_shape())}}));
                auto output_b = m.insert_instruction(
                    ins, make_op("gpu::int8_gemm_pack_a"), {inputs[1], packed_b});
                inputs[1] = output_b;
            }

            if(transa)
            {
                auto packed_a = m.insert_instruction(
                    ins, make_op("hip::allocate", {{"shape", to_value(inputs[0]->get_shape())}}));
                auto output_a = m.insert_instruction(
                    ins, make_op("gpu::int8_gemm_pack_b"), {inputs[0], packed_a});
                inputs[0] = output_a;
            }

            if(inputs != old_inputs)
            {
                m.replace_instruction(ins, ins->get_operator(), inputs);
            }
        }
        else if(ins->name() == "gpu::quant_convolution")
        {
            auto val = ins->get_operator().to_value();
            if(not val.at("int8_x4_format").to<bool>())
            {
                continue;
            }

            auto inputs   = ins->inputs();
            auto packed_x = m.insert_instruction(
                ins,
                make_op("hip::allocate",
                        {{"shape", to_value(pack_int8_shape(inputs[0]->get_shape()))}}));
            auto output_x =
                m.insert_instruction(ins, make_op("gpu::int8_conv_pack"), {inputs[0], packed_x});
            instruction::replace_argument(ins, inputs[0], output_x);

            auto packed_w = m.insert_instruction(
                ins,
                make_op("hip::allocate",
                        {{"shape", to_value(pack_int8_shape(inputs[1]->get_shape()))}}));
            auto output_w =
                m.insert_instruction(ins, make_op("gpu::int8_conv_pack"), {inputs[1], packed_w});
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

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
