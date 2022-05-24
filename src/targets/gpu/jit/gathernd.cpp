#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
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

// NOLINTNEXTLINE
static const char* const gathernd_kernel = R"__migraphx__(
#include <migraphx/kernels/gathernd.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

__global__ void gathernd_kernel(void* in_data, void* in_indices, void* output) 
{
    make_tensors()(in_data, in_indices, output)([](auto&&... xs) { 
        auto settings = make_gathernd_settings(MIGRAPHX_MAKE_CONSTANT(int64_t{BATCH_DIMS}));
        gathernd(xs..., settings); 
    });
}

}

} // namespace migraphx

)__migraphx__";

struct gathernd_compiler : compiler<gathernd_compiler>
{
    std::vector<std::string> names() const { return {"gathernd"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        auto out_s = inputs.back();
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "gathernd_kernel";
        options.virtual_inputs = inputs;

        // batch_dims
        assert(v.contains("batch_dims"));
        auto batch_dims = v.at("batch_dims").to<int64_t>();
        options.params += " -DBATCH_DIMS=" + std::to_string(batch_dims);

        return compile_hip_code_object(gathernd_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
