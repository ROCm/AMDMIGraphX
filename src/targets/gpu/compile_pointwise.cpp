#include <migraphx/gpu/compile_pointwise.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/context.hpp>
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

static const char* const pointwise_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
__global__ void kernel(${params}) 
{
    pointwise(${lambda}, ${args});
}
    
}

} // namespace migraphx

int main() {}

)__migraphx__";

operation compile_pointwise(context&,
                            const std::vector<shape>& inputs,
                            const std::string& lambda,
                            size_t global_workitems,
                            size_t local_workitems,
                            const std::string& preamble)
{
    hip_compile_options options;
    options.global         = global_workitems;
    options.local          = local_workitems;
    options.inputs         = inputs;
    options.output         = inputs.back();
    options.virtual_inputs = reduce_dims(inputs);
    options.params         = "-Wno-float-equal";
    auto src               = interpolate_string(pointwise_kernel,
                                  {{"params", enum_params(inputs.size(), "void * private_p")},
                                   {"args", enum_params(inputs.size(), "private_p")},
                                   {"lambda", lambda},
                                   {"preamble", preamble}});
    return compile_hip_code_object(src, options);
}
// Overload of compile_pointwise without global, local calculates global value at runtime
operation compile_pointwise(context& ctx,
                            const std::vector<shape>& inputs,
                            const std::string& lambda,
                            const std::string& preamble)
{
    assert(!inputs.empty());
    size_t local(1024);
    size_t n      = inputs.front().elements();
    size_t global = compute_global(n, local);
    return compile_pointwise(ctx, inputs, lambda, global, local, preamble);
}
operation compile_pointwise(context& ctx, const std::vector<shape>& inputs, module m)
{
    run_passes(m, {eliminate_common_subexpression{}, dead_code_elimination{}});
    cpp_generator g;
    g.fmap([](const std::string& fname) { return "migraphx::" + fname; });
    auto name = g.create_function(g.generate_module(m).set_attributes({"__device__"}));
    return compile_pointwise((ctx), inputs, "&" + name, g.str());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
