#ifndef MIGRAPHX_GUARD_RTGLIB_ELIMINATE_PAD_HPP
#define MIGRAPHX_GUARD_RTGLIB_ELIMINATE_PAD_HPP

#include <string>
#include <vector>
#include <array>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Remove pads if they can be written as an
 * attribute to another op (im2col, convolution, pooling)
 */
struct eliminate_pad
{
    std::string name() const { return "eliminate_pad"; }

    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
