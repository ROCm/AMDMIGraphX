#ifndef MIGRAPHX_GUARD_RTGLIB_COMPILE_HIP_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMPILE_HIP_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/compile_src.hpp>
#include <string>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

std::vector<std::vector<char>>
compile_hip_src(const std::vector<src_file>& srcs, std::string params, const std::string& arch);

std::string enum_params(std::size_t count, std::string param);

std::size_t compute_global(std::size_t n, std::size_t local = 1024);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
