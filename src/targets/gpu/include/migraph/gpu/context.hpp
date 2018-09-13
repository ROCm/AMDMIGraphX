#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/rocblas.hpp>
#include <migraph/gpu/hip.hpp>

#include <unordered_map>

namespace migraph {
namespace gpu {
using parameter_map = std::unordered_map<std::string, argument>;
struct context
{
    shared<miopen_handle> handle;
    shared<rocblas_handle_ptr> rbhandle;
    parameter_map params;
    argument scratch;
    std::vector<argument> literals{};
    void finish() const { gpu_sync(); }
};
} // namespace gpu
} // namespace migraph

#endif
