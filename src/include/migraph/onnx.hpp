#ifndef MIGRAPH_GUARD_MIGRAPHLIB_ONNX_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_ONNX_HPP

#include <migraph/program.hpp>
#include <migraph/config.hpp>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {

/// Create a program from an onnx file
program parse_onnx(const std::string& name);

} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
