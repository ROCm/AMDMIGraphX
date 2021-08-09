#ifndef MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_NHWC_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_NHWC_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Transform convolutions to nhwc
 */
struct layout_nhwc
{
    std::string name() const { return "layout_nhwc"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_NHWC_HPP
