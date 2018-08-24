#include <migraph/gpu/device/add_relu.hpp>
#include <migraph/gpu/device/nary.hpp>

namespace migraph {
namespace gpu {
namespace device {

void add_relu(argument result, argument arg1, argument arg2)
{
    nary_standard(result, arg1, arg2)([](auto x, auto y) { return max(0, x + y); });
}

} // namespace device
} // namespace gpu
} // namespace migraph
