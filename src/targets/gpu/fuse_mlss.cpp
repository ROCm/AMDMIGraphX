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
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/env.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/gpu/fuse_mlss.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/*
 * Comma-separated list of MLSS ops to enable, e.g. MIGRAPHX_MLSS_USE_SPECIFIC_OPS=mha
 * If unset, no MLSS ops are fused. Recognized values: "mha".
 */
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLSS_USE_SPECIFIC_OPS);

bool mlss_enabled()
{
    return not string_value_of(MIGRAPHX_MLSS_USE_SPECIFIC_OPS{}, "").empty();
}

static bool mlss_op_enabled(std::string_view op_name)
{
    const auto ops = split_string(string_value_of(MIGRAPHX_MLSS_USE_SPECIFIC_OPS{}, ""), ',');
    return std::any_of(ops.begin(), ops.end(), [&](const auto& opt) { return opt == op_name; });
}

struct mlss_mha
{
    std::string name() const { return "mlss_mha"; }

    std::optional<shape> output_shape = nullopt;

    shape compute_shape(std::vector<shape> inputs) const
    {
        if(output_shape.has_value())
            return output_shape.value();
        return inputs.back();
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
    if(not starts_with(ctx->get_current_device().get_gfx_name(), "gfx1201"))
        return;

    // Supported [batch, heads, seq, head_dim] shapes for the pre-compiled kernels.
    // Add new entries here to enable additional configurations.
    static const std::vector<std::vector<std::size_t>> supported_shapes = {
        {1, 8, 4096, 40},
    };

    if(mlss_op_enabled("mha"))
    {
        for(auto ins : iterator_for(m))
        {
            if(ins->name() != "group")
                continue;

            auto op_val = ins->get_operator().to_value();
            if(not op_val.contains("tag") or op_val["tag"].to<std::string>() != "attention")
                continue;

            auto& mod_args = ins->module_inputs();
            if(mod_args.empty())
                continue;

            module_ref attn_mod = mod_args[0];

            // Find the half-precision scale literal inside the submodule
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

            auto inputs = ins->inputs();
            if(inputs.size() != 3)
                continue;

            auto query_len = inputs[0]->get_shape().lens();
            bool shape_supported = std::any_of(
                supported_shapes.begin(), supported_shapes.end(), [&](const auto& s) {
                    return query_len == s;
                });

            if(not shape_supported)
                continue;

            // Hoist the scale literal into the parent module so it can be passed as an input
            auto scale_in_main = m.insert_literal(ins, scale_literal_ins->get_literal());

            m.replace_instruction(
                ins,
                make_op("mlss_mha", {{"output_shape", to_value(ins->get_shape())}}),
                {inputs[0], inputs[1], inputs[2], scale_in_main});
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
