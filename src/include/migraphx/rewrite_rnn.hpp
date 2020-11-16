#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_RNN_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_RNN_HPP

#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/config.hpp>
#include <migraphx/op/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;
using module = program;

/**
 * Rewrite rnn to gemm and add.
 */
struct rewrite_rnn
{
    std::string name() const { return "rewrite_rnn"; }
    void apply(module& prog) const;

    private:
    // for vanilla rnn operators
    void apply_vanilla_rnn(module& prog, instruction_ref ins) const;
    std::vector<instruction_ref> vanilla_rnn_cell(bool is_forward,
                                                  module& prog,
                                                  instruction_ref ins,
                                                  std::vector<instruction_ref> inputs,
                                                  operation& actv_func) const;
    std::vector<operation> vanilla_rnn_actv_funcs(instruction_ref ins) const;

    // for gru operators
    void apply_gru(module& prog, instruction_ref ins) const;
    std::vector<instruction_ref> gru_cell(bool is_forward,
                                          module& prog,
                                          instruction_ref ins,
                                          std::vector<instruction_ref> inputs,
                                          int linear_before_reset,
                                          const operation& actv_func1,
                                          const operation& actv_func2) const;

    std::vector<operation> gru_actv_funcs(instruction_ref ins) const;

    // for lstm operators
    void apply_lstm(module& prog, instruction_ref ins) const;
    std::vector<instruction_ref> lstm_cell(bool is_forward,
                                           module& prog,
                                           instruction_ref ins,
                                           std::vector<instruction_ref> inputs,
                                           const operation& actv_func1,
                                           const operation& actv_func2,
                                           const operation& actv_func3) const;

    std::vector<operation> lstm_actv_funcs(instruction_ref ins) const;

    bool is_variable_seq_lens(const module& prog, instruction_ref seq_lens) const;
    instruction_ref replace_last_hs_output(module& prog,
                                           instruction_ref ins,
                                           instruction_ref seq_lens,
                                           instruction_ref last_hs_output,
                                           op::rnn_direction dirct) const;

    void replace_last_cell_output(module& prog,
                                  instruction_ref ins,
                                  instruction_ref seq_lens,
                                  instruction_ref cell_outputs,
                                  instruction_ref last_cell_output,
                                  op::rnn_direction dirct) const;

    std::size_t
    get_seq_len(const module& prog, instruction_ref input, instruction_ref seq_lens) const;

    instruction_ref pad_hidden_states(module& prog,
                                      instruction_ref seq,
                                      instruction_ref seq_lens,
                                      instruction_ref hs) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
