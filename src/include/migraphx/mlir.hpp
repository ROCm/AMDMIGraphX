#ifndef MIGRAPHX_GUARD_RTGLIB_MLIR_HPP
#define MIGRAPHX_GUARD_RTGLIB_MLIR_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;

std::string dump_mlir(const module& m);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
