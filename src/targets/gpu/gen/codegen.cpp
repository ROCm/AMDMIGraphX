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
#include <migraphx/gpu/gen/codegen.hpp>
#include <migraphx/gpu/gen/tiling.hpp>
#include <migraphx/gpu/gen/gridwise.hpp>
#include <migraphx/gpu/gen/blockwise.hpp>
#include <migraphx/gpu/gen/lanewise.hpp>
#include <migraphx/gpu/gen/final.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/cpp_generator.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_COMPILE);

static const char* const gen_kernel_template = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/gen.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    auto idx = make_index();
    make_tensors()(${args})([&](auto... xs) {
        gen_func(xs..., idx);
    });
}

}

} // namespace migraphx

)__migraphx__";

std::string generate_gen_function(const module& m)
{
    cpp_generator g;
    g.fmap([](const std::string& fname) { return "migraphx::" + fname; });
    g.add_point_op("where", "${function:where}(${0}, ${1}, ${2})");
    g.add_point_op("prelu", "${function:where}(${0} < 0, ${0} * ${1}, ${0})");
    g.add_point_op("sign", "${function:where}(${0} > 0, 1, ${function:where}(${0} < 0, -1, 0))");
    g.add_point_op("equal", "migraphx::abs(${0} == ${1})");
    g.add_point_op("less", "migraphx::abs(${0} < ${1})");
    g.add_point_op("greater", "migraphx::abs(${0} > ${1})");
    g.add_point_op("not", "migraphx::abs(not ${0})");

    auto f = g.generate_module(m, [&](instruction_ref ins, const auto& names) -> std::string {
        auto v = ins->get_operator().attributes();
        if(v.contains("gpu_gen"))
        {
            auto code_template = v.at("gpu_gen").to<std::string>();
            auto args          = cpp_generator::to_args(ins->inputs(), names);
            std::string result = code_template;
            for(std::size_t i = 0; i < args.size(); ++i)
            {
                auto placeholder = "${" + std::to_string(i) + "}";
                auto pos         = result.find(placeholder);
                while(pos != std::string::npos)
                {
                    result.replace(pos, placeholder.size(), args[i]);
                    pos = result.find(placeholder, pos + args[i].size());
                }
            }
            return result;
        }
        if(v.contains("point_op"))
        {
            return g.generate_point_op(ins->get_operator(),
                                       cpp_generator::to_args(ins->inputs(), names));
        }
        if(ins->name() == "multibroadcast")
        {
            return names.at(ins->inputs().front());
        }
        // tile_region passes through to the tensor input
        if(ins->name() == "gpu::gen::tile_region")
        {
            return names.at(ins->inputs().front());
        }
        // lds_allocate: use a GCC statement expression to declare __shared__ and return pointer
        if(ins->name() == "gpu::gen::lds_allocate")
        {
            auto lds_shape = ins->get_shape();
            return "({static __shared__ " + shape::cpp_type(lds_shape.type()) + " lds_data[" +
                   std::to_string(lds_shape.elements()) + "]; lds_data;})";
        }
        MIGRAPHX_THROW("gen codegen: Unknown operator: " + ins->name());
    });

    f.set_attributes({"__device__"}).set_generic_types(m).set_name("gen_func");
    f.add_generic_param("idx");
    g.create_function(f);
    return g.str();
}

std::string
generate_gen_kernel(const module& m, const std::string& kernel_name, std::size_t total_params)
{
    auto preamble = generate_gen_function(m);

    return interpolate_string(gen_kernel_template,
                              {{"kernel", kernel_name},
                               {"params", enum_params(total_params, "void * private_p")},
                               {"args", enum_params(total_params, "private_p")},
                               {"preamble", preamble}});
}

operation compile_gen(context& ctx, const std::vector<shape>& in_shapes, const_module_ref pm)
{
    module m = *pm;

    // Run the 4-level lowering pipeline directly to avoid intermediate
    // shape validation that run_passes performs between passes.
    gen_gridwise{}.apply(m);
    gen_blockwise{}.apply(m);
    gen_lanewise{}.apply(m);
    gen_final{}.apply(m);

    // Remove dead instructions left from lowering
    run_passes(m, {dead_code_elimination{}});
    m.sort();

    if(enabled(MIGRAPHX_TRACE_COMPILE{}))
    {
        std::cerr << "compile_gen: lowered module:\n" << m << std::endl;
        std::cerr << "compile_gen: in_shapes=" << in_shapes.size()
                  << " module_params=" << m.get_parameter_shapes().size() << std::endl;
    }

    // Compute tiling config
    auto config = compute_tile_config(in_shapes);

    // Generate kernel source.
    // total_params = module params (inputs + z_output) which matches in_shapes
    auto kernel_name  = generate_name_from_ops(*pm, "gen_kernel");
    auto total_params = m.get_parameter_shapes().size();
    auto src          = generate_gen_kernel(m, kernel_name, total_params);

    if(enabled(MIGRAPHX_TRACE_COMPILE{}))
        std::cerr << "compile_gen: kernel source:\n" << src << std::endl;

    // Set up compile options
    hip_compile_options options;
    options.inputs      = in_shapes;
    options.output      = in_shapes.back();
    options.kernel_name = kernel_name;
    options.emplace_param("-Wno-float-equal");
    options.emplace_param("-Wno-unused-parameter");
    options.emplace_param("-Wno-unused-variable");
    options.emplace_param("-Wno-gnu-statement-expression");

    // Detect reduction ops in the lowered module for launch param selection
    bool has_reduce       = std::any_of(m.begin(), m.end(), [](const auto& ins) {
        return ins.name() == "gpu::gen::dpp_reduce" or ins.name() == "gpu::gen::reduce_waves" or
               ins.name() == "gpu::gen::lane_reduce";
    });
    bool has_block_reduce = std::any_of(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "gpu::gen::reduce_waves"; });

    if(has_reduce)
    {
        auto output_elements = options.output.elements();
        if(has_block_reduce)
        {
            // Block reduce: one workgroup per output element
            std::size_t block_size = 256;
            options.set_launch_params(value{}, output_elements * block_size, block_size);
        }
        else
        {
            // Wave or lane reduce
            auto input_elements = in_shapes.front().elements();
            options.set_launch_params(value{}, compute_global_for(ctx, input_elements, 256));
        }
    }
    else if(config.ntiles > 0)
    {
        options.set_launch_params(
            value{},
            compute_global_for(ctx, config.grid_size * config.block_size, 256),
            config.block_size);
    }
    else
    {
        auto elements = options.output.elements();
        options.set_launch_params(value{}, compute_global_for(ctx, elements, 256));
    }

    return compile_hip_code_object(ctx, src, options);
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
