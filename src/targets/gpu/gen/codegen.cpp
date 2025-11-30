/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/gen/codegen.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/cpp_generator.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

// Use gpu::gen namespace for compile_gen utilities
// Note: tile here refers to the compile_gen tile struct, not our tile_region operation
using migraphx::gpu::gen::find_fast_axis;
using migraphx::gpu::gen::generate_name_from_ops;
using migraphx::gpu::gen::generate_pointwise;
using migraphx::gpu::gen::make_transformer_args;
using migraphx::gpu::gen::tile;
using migraphx::gpu::gen::vectorize;

// Kernel template for gen pointwise operations (used by gen_pointwise_compiler)
static const char* const gen_pointwise_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
MIGRAPHX_GLOBAL void ${kernel}(${params}) 
{
    auto idx = make_index();
    pointwise<${noutputs}, ${tiled}>(idx, ${transformers})(${lambda}, ${args});
}
    
}

} // namespace migraphx

)__migraphx__";

// Kernel template for lowered gen IR with explicit vector_load/vector_store
static const char* const gen_lowered_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/debug.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"

MIGRAPHX_GLOBAL void ${kernel}(${params}) 
{
    auto idx = make_index();
    make_tensors()(${args})([&](auto... tensors) {
        auto get_tensor = [&](auto i) {
            return std::get<decltype(i)::value>(std::tie(tensors...));
        };
        (void)idx;
        (void)get_tensor;
        ${body}
    });
}

#pragma clang diagnostic pop
    
}

} // namespace migraphx

)__migraphx__";

std::string generate_pointwise_kernel(const module& m, const std::string& kernel_name)
{
    // Use the existing generate_pointwise function from compile_gen
    return migraphx::gpu::gen::generate_pointwise(m, kernel_name, true);
}

// Generate C++ code for a gen IR module
std::string generate_gen_code(const module& m, const std::string& /* kernel_name */)
{
    std::unordered_map<instruction_ref, std::string> names;
    std::size_t idx = 0;

    // Assign names to parameters (they map to tensor indices)
    auto param_names = m.get_parameter_names();
    std::sort(param_names.begin(), param_names.end());
    for(const auto& name : param_names)
    {
        auto param   = m.get_parameter(name);
        names[param] = "get_tensor(_c<" + std::to_string(idx) + ">)";
        idx++;
    }

    std::ostringstream body;

    // Generate code for each instruction
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@param")
            continue;
        if(ins->name() == "@return")
            continue;

        std::string ins_name = "v" + std::to_string(idx++);
        names[ins]           = ins_name;

        auto get_arg = [&](instruction_ref arg) -> std::string {
            auto it = names.find(arg);
            if(it != names.end())
                return it->second;
            return "/* unknown */";
        };

        if(ins->name() == "gpu::gen::global_id")
        {
            body << "        auto " << ins_name << " = idx.global;\n";
        }
        else if(ins->name() == "gpu::gen::local_id")
        {
            body << "        auto " << ins_name << " = idx.local;\n";
        }
        else if(ins->name() == "gpu::gen::workgroup_id")
        {
            body << "        auto " << ins_name << " = idx.group;\n";
        }
        else if(ins->name() == "gpu::gen::workgroup_size")
        {
            body << "        auto " << ins_name << " = idx.nlocal();\n";
        }
        else if(ins->name() == "gpu::gen::lane_id")
        {
            body << "        auto " << ins_name << " = idx.local_wave();\n";
        }
        else if(ins->name() == "gpu::gen::vector_load")
        {
            auto tensor = get_arg(ins->inputs()[0]);
            auto index  = get_arg(ins->inputs()[1]);
            auto size   = ins->get_operator().to_value().at("size").to<std::size_t>();
            if(size <= 1)
            {
                // Scalar load
                body << "        auto " << ins_name << " = " << tensor << "[" << index << "];\n";
            }
            else
            {
                // Vector load: use as_vec to get vectorized pointer, then load
                body << "        auto " << ins_name << " = as_vec<" << size << ">(remove_bool("
                     << tensor << ".data()))[" << index << "];\n";
            }
        }
        else if(ins->name() == "gpu::gen::vector_store")
        {
            auto tensor = get_arg(ins->inputs()[0]);
            auto index  = get_arg(ins->inputs()[1]);
            auto data   = get_arg(ins->inputs()[2]);
            auto size   = ins->get_operator().to_value().at("size").to<std::size_t>();
            if(size <= 1)
            {
                // Scalar store
                body << "        " << tensor << "[" << index << "] = " << data << ";\n";
            }
            else
            {
                // Vector store: use as_vec to get vectorized pointer, then store
                body << "        as_vec<" << size << ">(remove_bool(" << tensor << ".data()))["
                     << index << "] = " << data << ";\n";
            }
        }
        else if(ins->name() == "gpu::gen::check")
        {
            auto cond = get_arg(ins->inputs()[0]);
            body << "        MIGRAPHX_CHECK(" << cond << ");\n";
        }
        else if(ins->name() == "gpu::gen::barrier")
        {
            body << "        __syncthreads();\n";
        }
        else if(ins->name() == "@literal")
        {
            // Generate literal value - simple scalar literals
            auto lit       = ins->get_literal();
            auto lit_shape = lit.get_shape();
            if(lit_shape.elements() == 1)
            {
                lit_shape.visit_type([&](auto as) {
                    body << "        auto " << ins_name << " = static_cast<"
                         << shape::cpp_type(lit_shape.type()) << ">(" << as.from(lit.data())
                         << ");\n";
                });
            }
        }
        else if(ins->name() == "pointwise")
        {
            // Handle inner pointwise operations
            if(not ins->module_inputs().empty())
            {
                auto* pm = ins->module_inputs().front();
                auto pf  = generate_pointwise(*pm, ins_name + "_inner", true);
                body << "        // Pointwise: " << ins_name << "\n";
                // Generate inline pointwise call
                body << "        auto " << ins_name << " = [&]() {\n";
                body << "            " << ins_name << "_inner(";
                bool first = true;
                for(auto arg : ins->inputs())
                {
                    if(not first)
                        body << ", ";
                    body << get_arg(arg);
                    first = false;
                }
                body << ");\n";
                body << "        }();\n";
            }
        }
        else
        {
            // For other operations, try to use point_op attribute
            auto attrs = ins->get_operator().attributes();
            if(attrs.contains("point_op"))
            {
                auto point_op = attrs.at("point_op").to<std::string>();
                // Simple substitution for ${N} placeholders
                std::string result = point_op;
                for(std::size_t i = 0; i < ins->inputs().size(); i++)
                {
                    result = replace_string(
                        result, "${" + std::to_string(i) + "}", get_arg(ins->inputs()[i]));
                }
                body << "        auto " << ins_name << " = " << result << ";\n";
            }
            else
            {
                // Skip unsupported operations
                body << "        // Unsupported: " << ins->name() << "\n";
            }
        }
    }

    return body.str();
}

// Compile a gen IR program to a GPU code object
operation compile_gen(context& ctx, const program& p, const std::string& kernel_name)
{
    const auto* mm = p.get_main_module();

    hip_compile_options options;
    options.kernel_name = kernel_name;
    options.emplace_param("-Wno-float-equal");

    // Collect parameter shapes for args.hpp generation
    auto param_names = mm->get_parameter_names();
    std::sort(param_names.begin(), param_names.end());
    for(const auto& name : param_names)
    {
        auto shape = mm->get_parameter_shape(name);
        options.inputs.push_back(shape);
    }

    // Determine output shape (last parameter by convention)
    if(not options.inputs.empty())
        options.output = options.inputs.back();
    else
        options.output = shape{};

    options.virtual_inputs = options.inputs;

    // For now, use simple launch params - can be optimized later
    std::size_t total_elements = 1;
    if(not options.inputs.empty())
        total_elements = options.inputs.front().elements();

    options.global = total_elements;
    options.local  = std::min<std::size_t>(256, total_elements);
    if(options.global == 0)
    {
        options.global = 1;
        options.local  = 1;
    }

    // Generate the kernel body
    std::string body = generate_gen_code(*mm, kernel_name);

    // Generate parameters and args
    std::string params;
    std::string args;
    for(std::size_t i = 0; i < options.inputs.size(); i++)
    {
        if(i > 0)
        {
            params += ", ";
            args += ", ";
        }
        params += "void* private_p" + std::to_string(i);
        args += "private_p" + std::to_string(i);
    }

    auto src = interpolate_string(gen_lowered_kernel,
                                  {{"kernel", options.kernel_name},
                                   {"params", params},
                                   {"args", args},
                                   {"preamble", ""},
                                   {"body", body}});

    return compile_hip_code_object(ctx, src, options);
}

/// Gen pointwise compiler - compiles gpu::gen::pointwise operations using gen IR
struct gen_pointwise_compiler : compiler<gen_pointwise_compiler>
{
    std::vector<std::string> names() const { return {"gpu::gen::pointwise"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs         = flatten(inputs);
        options.output         = inputs.back();
        options.virtual_inputs = reduce_dims(normalize_permutation(options.inputs));
        options.emplace_param("-Wno-float-equal");

        auto axis     = find_fast_axis(options.virtual_inputs);
        auto vec      = vectorize::elements(ctx, axis, options.virtual_inputs);
        auto noutputs = options.inputs.size() - inputs.size() + 1;
        auto t        = tile::elements(options.virtual_inputs, noutputs);

        options.kernel_name = v.get("kernel", "gen_pointwise_kernel");

        if(t.ntiles == 0)
            options.set_launch_params(
                v, compute_global_for(ctx, options.inputs.front().elements() / vec.size, 256));
        else
            options.set_launch_params(
                v, compute_global_for(ctx, t.ntiles * t.block_size, 256), t.block_size);

        auto src =
            interpolate_string(gen_pointwise_kernel,
                               {{"kernel", options.kernel_name},
                                {"params", enum_params(options.inputs.size(), "void * private_p")},
                                {"args", enum_params(options.inputs.size(), "private_p")},
                                {"lambda", v.at("lambda").to<std::string>()},
                                {"transformers", make_transformer_args(t, vec)},
                                {"tiled", t.ntiles > 0 ? "true" : "false"},
                                {"noutputs", std::to_string(noutputs)},
                                {"preamble", v.get("preamble", std::string{})}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation&) const
    {
        assert(not ins->module_inputs().empty());
        const_module_ref pm = ins->module_inputs().front();

        // Generate the pointwise preamble using gen IR
        auto pf            = generate_pointwise_kernel(*pm, "gen_inner_pointwise");
        std::string lambda = "MIGRAPHX_LIFT(gen_inner_pointwise)";
        auto kernel_name   = generate_name_from_ops(*pm, "gen_kernel");

        return compile_op(ctx,
                          to_shapes(ins->inputs()),
                          {{"lambda", lambda}, {"preamble", pf}, {"kernel", kernel_name}});
    }
};

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
