#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_RNN_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_RNN_HPP

#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
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
    // for vanilla rnn operators
    void apply_vanilla_rnn(program& prog, instruction_ref ins) const;
    std::vector<instruction_ref> vanilla_rnn_cell(bool is_forward,
                                                  program& prog,
                                                  instruction_ref ins,
                                                  instruction_ref input,
                                                  instruction_ref w,
                                                  instruction_ref r,
                                                  instruction_ref bias,
                                                  instruction_ref ih,
                                                  operation& actv_func) const;
    std::vector<operation> vanilla_rnn_actv_funcs(instruction_ref ins) const;

    // for gru operators
    void apply_gru(program& prog, instruction_ref ins) const;
    std::vector<instruction_ref> gru_cell(bool is_forward,
                                          program& prog,
                                          instruction_ref ins,
                                          std::vector<instruction_ref> inputs,
                                          int linear_before_reset,
                                          const operation& actv_func1,
                                          const operation& actv_func2) const;

    std::vector<operation> gru_actv_funcs(instruction_ref ins) const;

    // for lstm operators
    void apply_lstm(program& prog, instruction_ref ins) const;
    std::vector<instruction_ref> lstm_cell(bool is_forward,
                                           program& prog,
                                           instruction_ref ins,
                                           std::vector<instruction_ref> inputs,
                                           const operation& actv_func1,
                                           const operation& actv_func2,
                                           const operation& actv_func3) const;

    std::vector<operation> lstm_actv_funcs(instruction_ref ins) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
