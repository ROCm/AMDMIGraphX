#ifndef MIGRAPHX_GUARD_RTGLIB_PAD_REWRITE_HPP
#define MIGRAPHX_GUARD_RTGLIB_PAD_REWRITE_HPP

#include <string>
#include <vector>
#include <array>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Rewrite pads to use attribute from other instructions instead.
 */
struct pad_rewrite
{
    std::string name() const { return "pad_rewrite"; }
    void apply(program& p) const;
    template <class T>
    void update_op(T, instruction_ref ins, instruction_ref output, program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
