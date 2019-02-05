#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_GRU_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_GRU_HPP

#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Rewrite gru to gemm, mul, and add.
 */
struct rewrite_gru
{
    std::string name() const { return "rewrite_gru"; }
    void apply(program& prog) const;

    private:
    std::vector<instruction_ref> gru_cell(bool is_forward,
                                          program& prog,
                                          instruction_ref ins,
                                          instruction_ref input,
                                          instruction_ref w,
                                          instruction_ref r,
                                          instruction_ref bias,
                                          instruction_ref ih,
                                          int linear_before_reset,
                                          const operation& actv_func1,
                                          const operation& actv_func2) const;

    std::vector<operation> compute_actv_funcs(instruction_ref ins) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
