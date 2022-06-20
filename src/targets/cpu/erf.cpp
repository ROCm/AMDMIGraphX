#include <migraphx/config.hpp>
#include <migraphx/cpu/pointwise.hpp>
#include <migraphx/op/erf.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template struct cpu_unary<op::erf>;

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
