#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>

#include <migraphx/cpp_generator.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

static const char* const softmax_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/softmax.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {
__global__ void softmax_kernel(void* input_p, void* output_p) 
{
    transform_args(make_tensors(), ${transformers})(input_p, output_p)([](auto input, auto output) {
        softmax<${axis}>(input, output);
    });
}
    
}

} // namespace migraphx

)__migraphx__";

struct softmax_compiler : compiler<softmax_compiler>
{
    std::vector<std::string> names() const { return {"softmax"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // TODO: Use reduce_dims
        auto axis  = v.at("axis").to<int64_t>();
        auto faxis = find_fast_axis({inputs.front()});
        vectorize vec{};
        // Vectorize if the axis is a reduction axis
        if(inputs.back().lens()[faxis] == 1)
        {
            vec = vectorize::elements(faxis, inputs);
        }
        auto relements  = inputs[0].lens()[axis] / vec.size;
        auto nelements  = inputs.back().elements() / relements;
        auto block_size = compute_block_size(relements, 256);
        hip_compile_options options;
        options.set_launch_params(
            v, compute_global_for(ctx, nelements * block_size, 256), block_size);
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "softmax_kernel";

        auto src = interpolate_string(
            softmax_kernel,
            {{"transformers", make_transformer_args(vec)}, {"axis", to_string(axis)}});

        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
