#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>

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

static const char* const transposectx_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/transposectx.hpp>
#include <args.hpp>
namespace migraphx {
extern "C" {
__global__ void transposectx_kernel(void* input_p, void* output_p) 
{
    make_tensors()(input_p, output_p)([](auto input, auto output) {
        transposectx(input, output);
    });
}
    
}
} // namespace migraphx
)__migraphx__";

struct transposectx_compiler : compiler<transposectx_compiler>
{
    std::vector<std::string> names() const { return {"transposectx"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.set_launch_params(v, compute_global_for(ctx, inputs.back().elements()), inputs.front().lens().back());
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "transposectx_kernel";

        return compile_hip_code_object(transposectx_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

static const char* const transposeqkv_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/transposeqkv.hpp>
#include <args.hpp>
namespace migraphx {
extern "C" {
__global__ void transposeqkv_kernel(void* input_p, void* output_p) 
{
    make_tensors()(input_p, output_p)([](auto input, auto output) {
        transposeqkv(input, output);
    });
}
    
}
} // namespace migraphx
)__migraphx__";

struct transposeqkv_compiler : compiler<transposeqkv_compiler>
{
    std::vector<std::string> names() const { return {"transposeqkv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.set_launch_params(v, compute_global_for(ctx, inputs.back().elements()), inputs.front().lens().back());
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "transposeqkv_kernel";

        return compile_hip_code_object(transposeqkv_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
