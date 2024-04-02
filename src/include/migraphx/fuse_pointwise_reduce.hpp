#ifndef MIGRAPHX_GUARD_MIGRAPHX_FUSE_POINTWISE_REDUCE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_FUSE_POINTWISE_REDUCE_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

struct MIGRAPHX_EXPORT fuse_pointwise_reduce
{
    std::string name() const { return "fuse_pointwise_reduce"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_FUSE_POINTWISE_REDUCE_HPP
