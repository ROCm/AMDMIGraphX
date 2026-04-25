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
// #include <amdmlss/amdmlss_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct mlss_mha
{
    std::string name() const { return "mlss_mha"; }

    std::optional<shape> output_shape = nullopt;

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        if(output_shape.has_value())
            return output_shape.value();
    }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_shape, "output_shape"));
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
            if(!op_val.contains("tag") || op_val["tag"].to<std::string>() != "attention")
                continue;

            // get the submodule (attn0)
            auto& mod_args = ins->module_inputs();
            if(mod_args.empty())
                continue;

            module_ref attn_mod = mod_args[0];  // the "attn0" module

            // find the @literal (scale) inside the submodule
            instruction_ref scale_literal_ins = attn_mod->end();
            for(auto sub_ins : iterator_for(*attn_mod))
            {
                if(sub_ins->name() == "@literal")
                {
                    scale_literal_ins = sub_ins;
                    break;
                }
            }

            if(scale_literal_ins == attn_mod->end())
                continue;

            // read scale value from the submodule literal
            const char* scale_data = scale_literal_ins->get_literal().data();
            const half* scale_hp   = reinterpret_cast<const half*>(scale_data);
            float scale_val        = static_cast<float>(*scale_hp);

            // insert the literal into the parent module so it can be used as an input
            instruction_ref scale_in_main = m.insert_literal(ins, scale_literal_ins->get_literal());

            auto inputs = ins->inputs();

            if(inputs.size() == 3)
            {
                auto input_query = inputs[0];
                auto input_key   = inputs[1];
                auto input_value = inputs[2];

                shape query_shape = inputs[0]->get_shape();
                auto query_len    = query_shape.lens();

                if(query_len[0] == 1 && query_len[1] == 8 && query_len[2] == 4096 &&
                   query_len[3] == 40)
                {
                    const auto& device_name =
                        ctx == nullptr ? "" : ctx->get_current_device().get_gfx_name();

                    std::vector<instruction_ref> refs;
                    refs.push_back(input_query);
                    refs.push_back(input_key);
                    refs.push_back(input_value);
                    refs.push_back(scale_in_main);

                    m.replace_instruction(
                        ins,
                        make_op("mlss_mha", {{"output_shape", to_value(ins->get_shape())}}),
                        refs);
                }
            }
        }        
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx