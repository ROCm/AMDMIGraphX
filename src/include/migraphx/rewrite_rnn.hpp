#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_RNN_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_RNN_HPP

#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Rewrite rnn to gemm and add.
 */
struct rewrite_rnn
{
    std::string name() const { return "rewrite_rnn"; }
    void apply(program& prog) const;

    private:
    std::vector<instruction_ref> rnn_cell(bool is_forward,
                                          program& prog,
                                          instruction_ref ins,
                                          instruction_ref input,
                                          instruction_ref w,
                                          instruction_ref r,
                                          instruction_ref bias,
                                          instruction_ref ih,
                                          operation& actv_func) const;

    std::vector<operation> compute_actv_funcs(instruction_ref ins) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
