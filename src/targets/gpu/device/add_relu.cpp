#include <migraph/gpu/device/add_relu.hpp>
#include <migraph/gpu/device/nary.hpp>

namespace migraph {
namespace gpu {
namespace device {

void add_relu(argument result, argument arg1, argument arg2)
{
    nary(std::move(result), std::move(arg1), std::move(arg2))(
        [](auto x, auto y) { return std::max<decltype(x + y)>(0, x + y); });
}

} // namespace device
} // namespace gpu
} // namespace migraph
