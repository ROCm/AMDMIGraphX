/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <unordered_set>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/gpu/fuse_mlss.hpp>
#include <amdmlss/amdmlss_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct mlss_mha
{
    std::string name() const { return "mlss_mha"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        return inputs[0];
    }
};
MIGRAPHX_REGISTER_OP(mlss_mha);


void fuse_mlss::apply(module& m) const
{
    
    for(auto ins : iterator_for(m))
    {
        auto name = ins->name();

        if (name == "group")
        {
            auto op_val = ins->get_operator().to_value();
            
            if ( !op_val.contains("tag") || op_val["tag"].to<value::literal_to_string<std::string>>() != "attention")
            {
                continue;
            }

            auto inputs = ins->inputs();
            if (inputs.size() != 4)
            {
                continue;
            }
            

            auto input_query = inputs[0];
            auto input_key = inputs[1];
            auto input_scale = inputs[2];
            auto input_value = inputs[3];

            shape::type_t type = ins->get_shape().type();

            auto input_scale_inputs = input_scale->inputs();
            auto scale_literal = input_scale_inputs[0];
            
            // Scale andd device name here in case we want to call mlss get caps in this function
            const float* scale_f = nullptr;
            if(scale_literal->name() == "@literal")
            {
                const char* scale = scale_literal->get_literal().data();
                scale_f = reinterpret_cast<const float*>(scale);
            }
            const auto& device_name = ctx == nullptr ? "" : ctx->get_current_device().get_gfx_name();

            instruction_ref output = m.insert_instruction(ins, make_op("allocate", {{"shape", to_value(ins->get_shape())}}));

            std::vector<instruction_ref> refs;
            refs.push_back(input_query);
            refs.push_back(input_key);
            refs.push_back(input_value);
            refs.push_back(scale_literal);
            refs.push_back(output);

            m.replace_instruction(
                ins,
                make_op("gpu::precompile_op", 
                        {{"op", to_value(make_op("mlss_mha"))},
                         {"output_shape", to_value(ins->get_shape())}}),
                        refs);            
        }        
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
