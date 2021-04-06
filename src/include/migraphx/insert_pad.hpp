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

    void apply(module& p) const;
    void update_op(const instruction_ref& input, const instruction_ref& ins, module& p) const;

    void update_pooling(const instruction_ref& input, const instruction_ref& ins, module& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
