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
// #include <migraphx/ranges.hpp>
// #include <migraphx/auto_any_cast.hpp>
// #include <migraphx/value.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/gpu/fuse_mlss.hpp>


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
        
        auto inputs = ins->inputs();
        auto outputs = ins->outputs();
        if(inputs.empty())
            continue;
        auto name = ins->name();        

        if (name == "mlss_mha")
        {
            std::cout << "fuse_mlss ins name: " << name << std::endl;
            auto names    = m.get_parameter_names();
            for(std::size_t i = 0; i < inputs.size(); i++)
            {
                auto input_name = inputs[i]->name();
                auto input_shape = inputs[i]->get_shape();
            }

            instruction_ref output = m.insert_instruction(ins, make_op("allocate", {{"shape", to_value(ins->get_shape())}}));
            
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            m.replace_instruction(
                ins,
                make_op("gpu::precompile_op", {{"op", to_value(ins->get_operator())}}),
                refs);            
        }        
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
