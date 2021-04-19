#ifndef MIGRAPHX_GUARD_RTGLIB_INLINE_SUBGRAPH_HPP
#define MIGRAPHX_GUARD_RTGLIB_INLINE_SUBGRAPH_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct inline_subgraph
{
    std::string name() const { return "inline_subgraph"; }
    void apply(module& p) const;
    void inline_submodule(module& p, instruction_ref ins) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
