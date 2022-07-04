#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>

#include <migraphx/cpp_generator.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

static const char* const layernorm_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/layernorm.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <migraphx/kernels/preload.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
__global__ void ${kernel}(${params}) 
{
    auto idx = make_index();
    transform_args(make_tensors(), rotate_last(), ${transformers})(${args})([](auto... xs) {
        ${layernorm}<${axis}>(${post}, xs...);
    });
}
    
}

} // namespace migraphx

)__migraphx__";

struct layernorm_compiler : compiler<layernorm_compiler>
{
    std::vector<std::string> names() const
    {
        return {"layernorm", "gpu::prelayernorm", "gpu::preadd_layernorm"};
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // TODO: Use reduce_dims
        auto axis  = inputs.front().lens().size() - 1;
        auto faxis = find_fast_axis({inputs.front()});
        vectorize vec{};
        // Vectorize if the axis is a reduction axis
        if(axis == faxis)
        {
            vec = vectorize::elements(faxis, inputs);
        }
        auto preloads   = preload::broadcasts(axis, inputs);
        auto relements  = inputs[0].lens()[axis] / vec.size;
        auto nelements  = (inputs.back().elements() / inputs[0].lens()[axis]);
        auto block_size = compute_block_size(relements, 256);
        hip_compile_options options;
        options.set_launch_params(
            v, compute_global_for(ctx, nelements * block_size, 256), block_size);
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = v.get("kernel", "layernorm_kernel");

        auto src = interpolate_string(layernorm_kernel,
                                      {{"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"transformers", make_transformer_args(preloads, vec)},
                                       {"post", v.get("post", std::string{"op::id{}"})},
                                       {"preamble", v.get("preamble", std::string{})},
                                       {"layernorm", v.get("layernorm", std::string{"layernorm"})},
                                       {"axis", to_string(axis)}});

        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v         = op.to_value();
        v["layernorm"] = "layernorm";
        v["kernel"]    = "layernorm_kernel";
        if(op.name() == "gpu::preadd_layernorm")
        {
            v["layernorm"] = "add_layernorm";
            v["kernel"]    = "add_layernorm_kernel";
        }
        if(not ins->module_inputs().empty())
        {
            auto* pm      = ins->module_inputs().front();
            v["preamble"] = generate_pointwise(*pm, "post_layernorm");
            v["post"]     = "MIGRAPHX_LIFT(post_layernorm)";
            v["kernel"] =
                v["layernorm"].to<std::string>() + "_" + generate_name_from_ops(*pm) + "_kernel";
        }
        return replace(compile_op(ctx, to_shapes(ins->inputs()), v));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
