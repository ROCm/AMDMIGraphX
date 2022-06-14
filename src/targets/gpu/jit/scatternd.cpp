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
static const char* const scatternd_kernel = R"__migraphx__(
#include <migraphx/kernels/scatternd.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

__global__ void scatternd_kernel(void* in_indices, void* in_updates, void* output) 
{
    make_tensors()(in_indices, in_updates, output)([](auto&&... xs) { 
        scatternd(xs..., ${reduction}{}); 
    });
}

}

} // namespace migraphx

)__migraphx__";

struct scatternd_compiler : compiler<scatternd_compiler>
{
    std::vector<std::string> names() const
    {
        return {"scatternd_none", "scatternd_add", "scatternd_mul"};
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.set_launch_params(v, compute_global_for(ctx, inputs.at(1).elements()));
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = "scatternd_kernel";
        options.virtual_inputs = inputs;
        auto reduction         = "assign_" + v.get("reduction", std::string{"none"});
        auto src               = interpolate_string(scatternd_kernel, {{"reduction", reduction}});
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        assert(starts_with(op.name(), "scatternd_"));
        auto reduction = op.name().substr(10);
        return insert(compile_op(ctx,
                                 to_shapes({ins->inputs().begin() + 1, ins->inputs().end()}),
                                 {{"reduction", reduction}}));
    }

    compiler_replace insert(const operation& op) const
    {
        return [=](module& m, instruction_ref ins) {
            auto args = ins->inputs();
            args.back() =
                m.insert_instruction(ins, make_op("hip::copy"), args.front(), args.back());
            args.erase(args.begin());
            return m.replace_instruction(ins, op, args);
        };
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
