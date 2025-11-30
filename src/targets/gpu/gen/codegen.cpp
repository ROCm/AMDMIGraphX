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
using migraphx::gpu::gen::find_fast_axis;
using migraphx::gpu::gen::generate_name_from_ops;
using migraphx::gpu::gen::generate_pointwise;
using migraphx::gpu::gen::make_transformer_args;
using migraphx::gpu::gen::tile;
using migraphx::gpu::gen::vectorize;

// Kernel template for gen pointwise operations
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

// Kernel template for lowered gen IR
static const char* const gen_lowered_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/debug.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <migraphx/kernels/gen.hpp>
#include <args.hpp>

namespace migraphx {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"

${preamble}

#pragma clang diagnostic pop

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params}) 
{
    auto idx = make_index();
    make_tensors()(${args})([&](auto... xs) {
        ${func_name}(xs..., idx);
    });
}
    
}

} // namespace migraphx

)__migraphx__";

std::string generate_pointwise_kernel(const module& m, const std::string& kernel_name)
{
    return migraphx::gpu::gen::generate_pointwise(m, kernel_name, true);
}

// Helper to generate shape expression
static std::string generate_shape_expr(const shape& s)
{
    return "make_shape(" + generate_index_ints(s.lens()) + ", " + generate_index_ints(s.strides()) +
           ")";
}

// Generate code for a single gen IR instruction
static std::string generate_gen_instruction(cpp_generator& g,
                                            instruction_ref ins,
                                            const std::vector<std::string>& args)
{
    // Try to use cpp_generator for operations with point_op
    try
    {
        return g.generate_point_op(ins->get_operator(), args);
    }
    catch(...)
    {
        // Fall through to manual handling
    }

    // Handle operations that need special code generation
    if(ins->name() == "gpu::gen::offset")
    {
        auto v = ins->get_operator().to_value();
        auto s = from_value<shape>(v.at("shape"));
        return "gen::compute_offset(" + generate_shape_expr(s) + ", " + args[0] + ")";
    }
    if(ins->name() == "gpu::gen::shape_index")
    {
        auto v = ins->get_operator().to_value();
        auto s = from_value<shape>(v.at("input_shape"));
        return "gen::compute_offset(" + generate_shape_expr(s) + ", " + args[0] + ")";
    }
    if(ins->name() == "gpu::gen::vector_load")
    {
        auto size = ins->get_operator().to_value().at("size").to<std::size_t>();
        if(size <= 1)
            return args[0] + ".data()[" + args[1] + "]";
        return "gen::vec_load<" + std::to_string(size) + ">(" + args[0] + ".data(), " + args[1] +
               ")";
    }
    if(ins->name() == "gpu::gen::vector_store")
    {
        auto size = ins->get_operator().to_value().at("size").to<std::size_t>();
        if(size <= 1)
            return "(void)(" + args[0] + ".data()[" + args[1] + "] = " + args[2] + ")";
        return "(void)gen::vec_store<" + std::to_string(size) + ">(" + args[0] + ".data(), " +
               args[1] + ", " + args[2] + ")";
    }
    if(ins->name() == "gpu::gen::tile_region")
    {
        auto v                = ins->get_operator().to_value();
        auto tile_dims        = v.at("tile_dims").to_vector<std::size_t>();
        std::size_t tile_size = 1;
        for(auto d : tile_dims)
            tile_size *= d;
        return "make_tensor_view(" + args[0] + ".data() + " + args[1] + " * " +
               std::to_string(tile_size) + ", " + args[0] + ".get_shape())";
    }
    if(ins->name() == "gpu::gen::lds_allocate")
    {
        auto s = ins->get_shape();
        return "/* lds_allocate: " + std::to_string(s.element_space()) + " */";
    }
    if(ins->name() == "gpu::gen::pad_index")
    {
        auto v    = ins->get_operator().to_value();
        auto pads = v.at("pads").to_vector<std::size_t>();
        auto s    = from_value<shape>(v.at("input_shape"));
        return "gen::pad_index(" + generate_shape_expr(s) + ", " + generate_index_ints(pads) +
               ", " + args[0] + ")";
    }
    if(ins->name() == "gpu::gen::reverse_index")
    {
        auto v    = ins->get_operator().to_value();
        auto axes = v.at("axes").to_vector<std::size_t>();
        auto s    = from_value<shape>(v.at("input_shape"));
        return "gen::reverse_index(" + generate_shape_expr(s) + ", " + generate_index_ints(axes) +
               ", " + args[0] + ")";
    }
    if(ins->name() == "gpu::gen::gather_index")
    {
        auto v    = ins->get_operator().to_value();
        auto axis = v.at("axis").to<std::size_t>();
        auto s    = from_value<shape>(v.at("input_shape"));
        return "gen::gather_index<decltype(" + generate_shape_expr(s) + "), decltype(" + args[1] +
               "), " + std::to_string(axis) + ">(" + generate_shape_expr(s) + ", " + args[1] +
               ", " + args[0] + ")";
    }
    if(ins->name() == "gpu::gen::conditional_load")
    {
        auto size = ins->get_operator().to_value().at("size").to<std::size_t>();
        if(size <= 1)
            return "gen::conditional_load(" + args[0] + ", " + args[1] + ", " + args[2] + ")";
        return "(" + args[1] + " >= 0) ? gen::vec_load<" + std::to_string(size) + ">(" + args[0] +
               ".data(), " + args[1] + ") : " + args[2];
    }
    if(ins->name() == "pointwise")
    {
        if(not ins->module_inputs().empty())
        {
            auto* pm = ins->module_inputs().front();
            return to_c_id(pm->name()) + "(" + join_strings(args, ", ") + ")";
        }
    }

    return "/* unsupported: " + ins->name() + " */";
}

// Generate a function for a gen IR module using cpp_generator
static void generate_gen_function(cpp_generator& g, const module& m, const std::string& func_name)
{
    // Register gen IR operations as point_ops
    g.add_point_op("gpu::gen::global_id", "idx.global");
    g.add_point_op("gpu::gen::local_id", "idx.local");
    g.add_point_op("gpu::gen::workgroup_id", "idx.group");
    g.add_point_op("gpu::gen::workgroup_size", "idx.nlocal()");
    g.add_point_op("gpu::gen::lane_id", "idx.local_wave()");
    g.add_point_op("gpu::gen::barrier", "(void)__syncthreads()");
    g.add_point_op("gpu::gen::check", "(void)MIGRAPHX_CHECK(${0})");

    // Use generate_module with custom callback
    auto f = g.generate_module(m, [&](instruction_ref ins, const auto& names) {
        return generate_gen_instruction(g, ins, cpp_generator::to_args(ins->inputs(), names));
    });

    // Set function name and attributes - idx is passed first, then tensors
    f.set_name(func_name).set_generic_types(m).add_generic_param("idx").set_attributes(
        {"__device__"});

    // Create the function (writes to g's internal stream)
    g.create_function(f);
}

// Compile a gen IR program to a GPU code object
operation compile_gen(context& ctx, const program& p, const std::string& kernel_name)
{
    const auto* mm = p.get_main_module();

    hip_compile_options options;
    options.kernel_name = kernel_name;
    options.emplace_param("-Wno-float-equal");

    auto param_names = mm->get_parameter_names();
    std::sort(param_names.begin(), param_names.end());
    for(const auto& name : param_names)
    {
        options.inputs.push_back(mm->get_parameter_shape(name));
    }

    if(not options.inputs.empty())
        options.output = options.inputs.back();
    else
        options.output = shape{};

    options.virtual_inputs = options.inputs;

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

    // Generate the function using cpp_generator
    cpp_generator g;
    std::string func_name = "gen_func";
    generate_gen_function(g, *mm, func_name);
    std::string preamble = g.str();

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
                                   {"preamble", preamble},
                                   {"func_name", func_name}});

    return compile_hip_code_object(ctx, src, options);
}

/// Gen pointwise compiler
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
