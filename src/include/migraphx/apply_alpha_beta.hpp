#ifndef MIGRAPHX_GUARD_MIGRAPHX_APPLY_ALPHA_BETA_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_APPLY_ALPHA_BETA_HPP

#include <migraphx/instruction_ref.hpp>
#include <migraphx/module.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

instruction_ref insert_apply_alpha_beta(module& m,
                                        instruction_ref pos,
                                        const std::vector<instruction_ref>& args,
                                        const std::string& op_name,
                                        const literal& alpha,
                                        const literal& beta);

template <typename T = float>
instruction_ref insert_apply_alpha_beta(module& m,
                                        instruction_ref pos,
                                        const std::vector<instruction_ref>& args,
                                        const std::string& op_name,
                                        T alpha = 1.0f,
                                        T beta  = 0.0f)
{
    return insert_apply_alpha_beta(m, pos, args, op_name, literal{T{alpha}}, literal{T{beta}});
}

template <typename T = float>
instruction_ref add_apply_alpha_beta(module& m,
                                     const std::vector<instruction_ref>& args,
                                     const std::string& op_name,
                                     T alpha = 1.0f,
                                     T beta  = 0.0f)
{
    return insert_apply_alpha_beta(m, m.end(), args, op_name, alpha, beta);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_APPLY_ALPHA_BETA_HPP
