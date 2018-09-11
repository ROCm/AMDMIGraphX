#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

#include <migraph/argument.hpp>
#include <unordered_map>

namespace migraph {
namespace cpu {
using parameter_map = std::unordered_map<std::string, argument>;
    
struct context
{
    parameter_map params;
    void finish() const {}
};

} // namespace cpu
} // namespace migraph

#endif
