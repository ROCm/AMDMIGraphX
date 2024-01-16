#ifndef GUARD_SRC_INCLUDE_MIGRAPHX_REWRITE_REDUCE
#define GUARD_SRC_INCLUDE_MIGRAPHX_REWRITE_REDUCE

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct MIGRAPHX_EXPORT rewrite_reduce
{
    std::string name() const { return "rewrite_reduce"; }
    void apply(module& m) const;
};



} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif

