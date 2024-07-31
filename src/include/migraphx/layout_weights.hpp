#ifndef MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_WEIGHTS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_WEIGHTS_HPP

#include <migraphx/config.hpp>
#include <string>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct MIGRAPHX_EXPORT layout_weights
{
    std::string name() const { return "layout_weights"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_WEIGHTS_HPP
