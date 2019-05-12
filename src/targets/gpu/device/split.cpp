#include <hip/hip_runtime.h>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/split.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

__global__ void split_kernel(char* input,
                             const std::size_t* map,
                             char* output,
                             std::size_t n,
                             std::size_t bytes,
                             unsigned offset)
{
    unsigned global_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(global_id < n)
    {
        std::size_t map_index = map[global_id + offset];
        char* output_addr     = output + bytes * global_id;
        char* input_addr      = input + bytes * map_index;
        std::copy(input_addr, input_addr + bytes, output_addr);
    }
}

argument split(hipStream_t stream,
               const migraphx::shape& output_shape,
               std::vector<migraphx::argument> args,
               unsigned offset)
{
    auto result            = args.back();
    auto arg0              = args[0];
    auto arg1              = args[1];
    std::size_t nelements  = output_shape.elements();
    auto* output           = result.data();
    auto* input            = arg0.data();
    const std::size_t* map = reinterpret_cast<const std::size_t*>(arg1.data());
    std::size_t local      = 1024;
    std::size_t groups     = 1 + nelements / local;
    std::size_t nglobal    = std::min<std::size_t>(256, groups) * local;

    hipLaunchKernelGGL(split_kernel,
                       dim3(nglobal / local),
                       dim3(local),
                       0,
                       stream,
                       input,
                       map,
                       output,
                       nelements,
                       output_shape.type_size(),
                       offset);

    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
