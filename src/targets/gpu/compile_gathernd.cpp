#include <migraphx/gpu/compile_gathernd.hpp>
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
static const char* const gathernd_kernel = R"__migraphx__(
#include <migraphx/kernels/gathernd.hpp>
#include <migraphx/kernels/basic_ops.hpp>
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

int main() {}

)__migraphx__";

operation compile_gathernd(context&, const std::vector<shape>& io_shapes, const value& val)
{
    hip_compile_options options;
    auto out_s             = io_shapes.back();
    options.local          = 1024;
    options.global         = compute_global(out_s.elements(), options.local);
    options.inputs         = io_shapes;
    options.output         = out_s;
    options.kernel_name    = "gathernd_kernel";
    options.virtual_inputs = io_shapes;

    // batch_dims
    assert(val.contains("batch_dims"));
    auto batch_dims = val.at("batch_dims").to<int64_t>();
    options.params += " -DBATCH_DIMS=" + std::to_string(batch_dims);

    return compile_hip_code_object(gathernd_kernel, options);
}
} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
