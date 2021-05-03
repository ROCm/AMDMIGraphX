#ifndef MIGRAPHX_GUARD_RTGLIB_MIOPEN_MLIR_CONV_HPP
#define MIGRAPHX_GUARD_RTGLIB_MIOPEN_MLIR_CONV_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {
struct mlir_conv
{
    context* ctx;
    std::string name() const { return "mlir::convolution"; }
    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
