#ifndef MIGRAPHX_GUARD_MIGRAPHX_REWRITE_TOPK_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_REWRITE_TOPK_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct rewrite_topk
{
    std::string name() const { return "rewrite_topk"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_REWRITE_TOPK_HPP
