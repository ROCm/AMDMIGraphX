#include <migraph/gpu/device/contiguous.hpp>
#include <migraph/gpu/device/binary.hpp>

namespace migraph {
namespace gpu {
namespace device {

void add_relu(argument arg1, argument arg2, argument result)
{
    binary_standard(arg1, arg2, result, [](auto x, auto y) { 
        return max(0, x + y);
    });
}

} // namespace device
} // namespace gpu
} // namespace migraph
