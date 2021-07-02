#ifndef MIGRAPHX_GUARD_RTGLIB_INSERT_PAD_HPP
#define MIGRAPHX_GUARD_RTGLIB_INSERT_PAD_HPP

#include <string>
#include <vector>
#include <array>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * insert pads if attribute of padding is asymmetrical
 */
struct insert_pad
{
    std::string name() const { return "insert_pad"; }

    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
