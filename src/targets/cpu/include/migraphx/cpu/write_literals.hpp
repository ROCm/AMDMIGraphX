#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_WRITE_LITERALS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_WRITE_LITERALS_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;
namespace cpu {

struct write_literals
{
    std::string name() const { return "cpu::write_literals"; }
    void apply(module& m) const;
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
