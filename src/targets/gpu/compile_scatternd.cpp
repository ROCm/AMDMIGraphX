#include <migraphx/gpu/compile_scatternd.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const scatternd_kernel = R"__migraphx__(
#include <migraphx/kernels/scatternd.hpp>
#include <migraphx/kernels/basic_ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

__global__ void scatternd_kernel(void* in_indices, void* in_updates, void* output) 
{
    make_tensors()(in_indices, in_updates, output)([](auto&&... xs) { 
        scatternd(xs..., REDUCTION); 
    });
}

}

} // namespace migraphx

int main() {}

)__migraphx__";

operation
compile_scatternd(context&, const std::vector<shape>& io_shapes, const std::string& reduction)
{
    hip_compile_options options;
    auto out_s             = io_shapes.back();
    options.local          = 1024;
    options.global         = compute_global(io_shapes.at(1).elements(), options.local);
    options.inputs         = io_shapes;
    options.output         = out_s;
    options.kernel_name    = "scatternd_kernel";
    options.virtual_inputs = io_shapes;

    options.params += " -DREDUCTION=assign_" + reduction + "{}";

    return compile_hip_code_object(scatternd_kernel, options);
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
