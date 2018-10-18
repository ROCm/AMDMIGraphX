#include <migraph/gpu/device/add_relu.hpp>
#include <migraph/gpu/device/nary.hpp>

namespace migraph {
namespace gpu {
namespace device {

void add_relu(const argument& result, const argument& arg1, const argument& arg2)
{
    nary(result, arg1, arg2)([](auto x, auto y) { return std::max<decltype(x + y)>(0, x + y); });
}

void add_relu(const argument& result,
              const argument& arg1,
              const argument& arg2,
              const argument& arg3)
{
    nary(result, arg1, arg2, arg3)(
        [](auto x, auto y, auto z) { return std::max<decltype(x + y + z)>(0, x + y + z); });
}

} // namespace device
} // namespace gpu
} // namespace migraph
