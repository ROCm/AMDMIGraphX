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
 * Remove identity instructions. Currently when used as the last pass, it will
 * preserve the semantics of previous program state, therefore dead code elimination
 * should not be used afterwards.
 */
struct pad_rewrite
{
    std::string name() const { return "pad_rewrite"; }
    void apply(program& p) const;
    template <class T>
    void update_op(T, const instruction_ref& input, const instruction_ref& ins, program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
