#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_TF_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_TF_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Create a program from a tf pb file (default is nhwc format)
// program parse_tf(const std::string& name, bool is_nhwc);

program parse_tf(const std::string& name, bool is_nhwc, unsigned int batch_size = 1);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
