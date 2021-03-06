#ifndef MIGRAPHX_GUARD_CPU_FUSE_OPS_HPP
#define MIGRAPHX_GUARD_CPU_FUSE_OPS_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace cpu {

struct fuse_ops
{
    std::string name() const { return "cpu::fuse_ops"; }
    void apply(module& m) const;
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_CPU_FUSE_OPS_HPP
