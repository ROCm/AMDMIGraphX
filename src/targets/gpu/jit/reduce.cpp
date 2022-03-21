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

static const char* const simple_reduce_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
__global__ void kernel(void* input_p, void* output_p) 
{
    make_tensors()(input_p, output_p)([](auto input, auto output) {

        simple_reduce(${reduction}, ${init}, input, output, ${read}, ${write});
    });
}
    
}

} // namespace migraphx

)__migraphx__";

struct reduce_compiler : compiler<reduce_compiler>
{
    std::vector<std::string> names() const { return {"reduce"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.set_launch_params(v, compute_global_for(ctx, inputs.back().elements()));
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = reduce_dims(inputs);
        options.params         = "-Wno-float-equal";
        std::string identity   = "[](auto x) { return x; }";
        auto src               = interpolate_string(simple_reduce_kernel,
                                      {{"reduction", v.at("reduction").to<std::string>()},
                                       {"init", v.get("init", std::string{"0"})},
                                       {"read", v.get("read", identity)},
                                       {"write", v.get("write", identity)},
                                       {"preamble", v.get("preamble", std::string{})}});
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation&) const
    {
        return {};
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
