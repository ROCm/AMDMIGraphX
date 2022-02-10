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

__global__ void scatternd_kernel(void* in_data, void* in_indices, void* in_updates, void* output) 
{
    make_tensors()(in_data, in_indices, in_updates, output)([](auto&&... xs) { 
        auto settings = make_scatternd_settings(_c<bool{IS_ADD}>, _c<bool{IS_MUL}>);
        scatternd(xs..., settings); 
    });
}

}

} // namespace migraphx

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
static const char* const scatternd_copy_kernel = R"__migraphx__(
#include <migraphx/kernels/scatternd.hpp>
#include <migraphx/kernels/basic_ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

__global__ void scatternd_copy_kernel(void* in_data, void* in_indices, void* in_updates, void* output) 
{
    make_tensors()(in_data, in_indices, in_updates, output)([](auto&&... xs) { 
        scatternd_copy(xs...); 
    });
}

}

} // namespace migraphx

int main() {}

)__migraphx__";

operation compile_scatternd(context&, const std::vector<shape>& io_shapes, const value& val)
{
    hip_compile_options options;
    auto out_s             = io_shapes.back();
    options.local          = 1024;
    options.global         = compute_global(io_shapes.at(2).elements(), options.local);
    options.inputs         = io_shapes;
    options.output         = out_s;
    options.kernel_name    = "scatternd_kernel";
    options.virtual_inputs = io_shapes;

    // reduction
    assert(val.contains("reduction"));
    auto reduction = val.at("reduction").to<std::string>();
    bool is_add    = reduction == "add";
    bool is_mul    = reduction == "mul";
    options.params += " -DIS_ADD=" + std::to_string(static_cast<int>(is_add));
    options.params += " -DIS_MUL=" + std::to_string(static_cast<int>(is_mul));

    return compile_hip_code_object(scatternd_kernel, options);
}

operation compile_scatternd_copy(context&, const std::vector<shape>& io_shapes)
{
    hip_compile_options options;
    auto out_s             = io_shapes.back();
    options.local          = 1024;
    options.global         = compute_global(out_s.elements(), options.local);
    options.inputs         = io_shapes;
    options.output         = out_s;
    options.kernel_name    = "scatternd_copy_kernel";
    options.virtual_inputs = io_shapes;

    return compile_hip_code_object(scatternd_copy_kernel, options);
}
} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
