#ifndef MIGRAPHX_GUARD_RTGLIB_CPU_LOWERING_HPP
#define MIGRAPHX_GUARD_RTGLIB_CPU_LOWERING_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace ref {

struct lowering
{
    std::string name() const { return "ref::lowering"; }
    void apply(module& m) const;
};

} // namespace ref
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
