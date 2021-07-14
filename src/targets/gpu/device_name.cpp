#include <migraphx/gpu/device_name.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/rank.hpp>
#include <migraphx/stringutils.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

template <class HipDeviceProp>
std::string get_arch_name(rank<0>, const HipDeviceProp& props)
{
    return "gfx" + std::to_string(props.gcnArch);
}

template <class HipDeviceProp>
auto get_arch_name(rank<1>, const HipDeviceProp& props) -> decltype(std::string(props.gcnArchName))
{
    return std::string(props.gcnArchName);
}

int get_device_id()
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        MIGRAPHX_THROW("No device");
    return device;
}

std::string get_device_name()
{
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, get_device_id());
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to get device properties");
    return get_arch_name(rank<1>{}, props);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
