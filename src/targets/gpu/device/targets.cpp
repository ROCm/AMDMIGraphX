
#include <migraphx/gpu/device/targets.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/errors.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

static std::vector<std::string> parse_targets() { return split_string(MIGRAPHX_GPU_TARGETS, ';'); }

const std::vector<std::string>& get_targets()
{
    static auto result = parse_targets();
    return result;
}

std::string get_targets_as_string() { return join_strings(get_targets(), ", "); }

static int get_device_id()
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
    return props.gcnArchName;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
