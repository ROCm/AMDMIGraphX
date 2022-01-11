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
                            const std::string& preamble)
{
    hip_compile_options options;
    options.global         = compute_global(inputs.front().elements());
    options.local          = 1024;
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

operation compile_pointwise(context& ctx, const std::vector<shape>& inputs, module m)
{
    run_passes(m, {eliminate_common_subexpression{}, dead_code_elimination{}});
    cpp_generator g;
    g.fmap([](const std::string& fname) { return "migraphx::" + fname; });
    g.add_point_op("where", "${function:where}(${0}, ${1}, ${2})");
    g.add_point_op("prelu", "${function:where}(${0} < 0, ${0} * ${1}, ${0})");
    g.add_point_op("sign", "${function:where}(${0} > 0, 1, ${function:where}(${0} < 0, -1, 0))");
    g.add_point_op("equal", "migraphx::abs(${0} == ${1})");
    g.add_point_op("less", "migraphx::abs(${0} < ${1})");
    g.add_point_op("greater", "migraphx::abs(${0} > ${1})");
    g.add_point_op("not", "migraphx::abs(not ${0})");
    auto name =
        g.create_function(g.generate_module(m).set_attributes({"__device__"}).set_generic_types(m));
    return compile_pointwise((ctx), inputs, "MIGRAPHX_LIFT(" + name + ")", g.str());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
