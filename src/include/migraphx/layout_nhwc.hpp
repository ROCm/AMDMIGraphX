#ifndef MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_NHWC_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_NHWC_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

/**
 * Transform convolutions to nhwc
 */
struct layout_nhwc
{
    std::string name() const { return "layout_nhwc"; }
    void apply(module_pass_manager& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_NHWC_HPP
