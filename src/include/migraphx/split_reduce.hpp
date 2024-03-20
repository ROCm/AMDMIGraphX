#ifndef MIGRAPHX_GUARD_MIGRAPHX_SPLIT_REDUCE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SPLIT_REDUCE_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

struct MIGRAPHX_EXPORT split_reduce
{
    std::size_t split_size = 8192;
    std::string name() const { return "split_reduce"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SPLIT_REDUCE_HPP
