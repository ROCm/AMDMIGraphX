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
#include <migraphx/kernels/basic_ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>
namespace migraphx {
extern "C" {
__global__ void transposectx_kernel(void* input_p, void* output_p) 
{
    make_tensors()(input_p, output_p)([](auto input, auto output) {
        auto settings = make_transposectx_settings(MIGRAPHX_MAKE_CONSTANT(int64_t{HEAD_SIZE}), MIGRAPHX_MAKE_CONSTANT(bool{REVERSED_BS}));
        transposectx(input, output, settings);
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
        options.set_launch_params(v, compute_global_for(ctx, inputs.back().elements()), 64);
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "transposectx_kernel";

        // head_size
        assert(v.contains("head_size"));
        auto head_size = v.at("head_size").to<int>();
        options.params += " -DHEAD_SIZE=" + std::to_string(head_size);

        // reversed_bs
        assert(v.contains("reversed_bs"));
        auto reversed_bs = v.at("reversed_bs").to<bool>();
        options.params += " -DREVERSED_BS=" + std::to_string(reversed_bs);

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
#include <migraphx/kernels/basic_ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>
namespace migraphx {
extern "C" {
__global__ void transposeqkv_kernel(void* input_p, void* output_p) 
{
    make_tensors()(input_p, output_p)([](auto input, auto output) {
        auto settings = make_transposeqkv_settings(MIGRAPHX_MAKE_CONSTANT(int64_t{HEAD_SIZE}), MIGRAPHX_MAKE_CONSTANT(bool{REVERSED_BS}));
        transposeqkv(input, output, settings);
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
        options.set_launch_params(v, compute_global_for(ctx, inputs.back().elements()), 64);
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "transposeqkv_kernel";

        // head_size
        assert(v.contains("head_size"));
        auto head_size = v.at("head_size").to<int>();
        options.params += " -DHEAD_SIZE=" + std::to_string(head_size);

        // reversed_bs
        assert(v.contains("reversed_bs"));
        auto reversed_bs = v.at("reversed_bs").to<bool>();
        options.params += " -DREVERSED_BS=" + std::to_string(reversed_bs);

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
