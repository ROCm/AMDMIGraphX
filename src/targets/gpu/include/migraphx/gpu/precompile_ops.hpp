/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef MIGRAPHX_GUARD_GPU_PRECOMPILE_OPS_HPP
#define MIGRAPHX_GUARD_GPU_PRECOMPILE_OPS_HPP

#include <migraphx/gpu/config.hpp>
#include <string>
#include <migraphx/operation.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct precompile_op
{
    operation op                      = op::identity{};
    std::size_t additional_args       = 1;
    bool ignore_modules               = false;
    std::optional<shape> output_shape = nullopt;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"),
                    f(self.additional_args, "additional_args"),
                    f(self.ignore_modules, "ignore_modules"),
                    f(self.output_shape, "output_shape"));
    }

    std::string name() const { return "gpu::precompile_op"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        // Pop off additional args
        inputs.resize(inputs.size() - additional_args);
        if(output_shape.has_value())
            return output_shape.value();
        if(ignore_modules)
            return op.compute_shape(inputs);
        return op.compute_shape(inputs, mods);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

struct dynamic_code_object_op
{
    operation pre_op                  = precompile_op{};
    std::optional<shape> output_shape = nullopt;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.pre_op, "pre_op"), f(self.output_shape, "output_shape"));
    }

    std::string name() const { return "gpu::dynamic_code_object_op"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        return pre_op.compute_shape(inputs, mods);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
    argument compute(context& ctx,
                     const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& module_args,
                     std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)> run) const
    {
        auto static_args = std::vector<argument>{args.begin(), args.end()};
        auto output_arg  = static_args.back();
        module static_mod;
        if(not module_args.empty())
        {
            // rewrite module without dynamic shapes
            auto mod_args = std::vector<argument>{args.begin(), args.end() - 1};
            static_mod    = module_args.front()->with_static_shapes(to_shapes(mod_args));
            static_mod.set_bypass(true);

            // compute output arg shape
            if(output_arg.get_shape().dynamic())
            {
                auto out_shapes = static_mod.compute_shapes(to_shapes(mod_args));
                auto rsp_shape  = (out_shapes.size() > 1) ? shape{out_shapes} : out_shapes.front();
                static_args[static_args.size() - 1] = output_arg.reshape(rsp_shape);
            }
        }
        else
        {
            if(output_arg.get_shape().dynamic())
            {
                auto out_shape                      = pre_op.compute_shape(to_shapes(static_args));
                static_args[static_args.size() - 1] = output_arg.reshape(out_shape);
            }
        }

        auto temp_mod = module("temp_mod");
        std::vector<instruction_ref> args_ins;
        std::vector<size_t> idx(static_args.size());
        std::iota(std::begin(idx), std::end(idx), 0);
        std::transform(static_args.begin(),
                       static_args.end(),
                       idx.begin(),
                       std::back_inserter(args_ins),
                       [&](const auto& arg, const auto& i) {
                           return temp_mod.add_parameter("temp_mod:x" + std::to_string(i),
                                                         arg.get_shape());
                       });
        instruction_ref ins;
        if(not module_args.empty())
        {
            ins = temp_mod.add_instruction(pre_op, args_ins, {&static_mod});
        }
        else
        {
            ins = temp_mod.add_instruction(pre_op, args_ins);
        }
        temp_mod.add_return({ins});

        operation preop = any_cast<precompile_op>(ins->get_operator()).op;
        auto config     = get_tuning_config(ctx, ins, preop, false);
        value solution  = value{};
        if(config.has_value())
        {
            solution = config->solutions.front();
        }
        auto compiled_op = compile(ctx, ins, preop, solution);
        compiled_op.replace(temp_mod, ins);
        run_passes(temp_mod, {dead_code_elimination{}});

        // Finalize the module before execution
        std::vector<migraphx::context> contexts = {migraphx::context(ctx)};
        temp_mod.finalize(contexts);

        // Build param_map based on ACTUAL parameters that exist
        auto param_map = std::unordered_map<std::string, argument>{};
        for(auto i : idx)
        {
            param_map["temp_mod:x" + std::to_string(i)] = static_args[i];
        }
        module_ref temp_mod_ref = &temp_mod;

        auto results = run(temp_mod_ref, param_map);

        if(results.size() > 1)
            return results;
        return results.front();
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_PRECOMPILE_OPS_HPP
