#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>
#include <migraphx/op/sub.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template struct cpu_binary<op::sub>;

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
