#include <migraphx/gpu/compile_pointwise.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static const char* const pointwise_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <args.hpp>

using namespace migraphx;

extern "C" {
__global__ void kernel(${params}) 
{
    pointwise(${lambda}, ${args});
}
    
}

int main() {}

)__migraphx__";

operation compile_pointwise(context&, const std::vector<shape>& inputs, const std::string& lambda)
{
    hip_compile_options options;
    options.global         = compute_global(inputs.front().elements());
    options.local          = 1024;
    options.inputs         = inputs;
    options.output         = inputs.back();
    options.reduced_inputs = reduce_dims(inputs);
    auto src               = interpolate_string(pointwise_kernel,
                                  {{"params", enum_params(inputs.size(), "void * private_p")},
                                   {"args", enum_params(inputs.size(), "private_p")},
                                   {"lambda", lambda}});
    return compile_hip_code_object(src, options);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
