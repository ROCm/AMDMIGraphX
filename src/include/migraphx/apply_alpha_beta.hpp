#ifndef MIGRAPHX_GUARD_MIGRAPHX_APPLY_ALPHA_BETA_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_APPLY_ALPHA_BETA_HPP

#include "migraphx/make_op.hpp"
#include "migraphx/normalize_attributes.hpp"
#include "migraphx/operation.hpp"
#include <migraphx/instruction_ref.hpp>
#include <migraphx/module.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

instruction_ref insert_apply_alpha_beta(module& m,
                                        instruction_ref pos,
                                        const std::vector<instruction_ref>& args,
                                        const operation& op,
                                        const literal& alpha,
                                        const literal& beta);

template <typename T = float>
instruction_ref insert_apply_alpha_beta(module& m,
                                        instruction_ref pos,
                                        const std::vector<instruction_ref>& args,
                                        const operation& op,
                                        T alpha = 1,
                                        T beta  = 0)
{
    return insert_apply_alpha_beta(m, pos, args, op, literal{T{alpha}}, literal{T{beta}});
}

template <typename T = float>
instruction_ref add_apply_alpha_beta(module& m,
                                     const std::vector<instruction_ref>& args,
                                     const operation& op,
                                     T alpha = 1,
                                     T beta  = 0)
{
    return insert_apply_alpha_beta(m, m.end(), args, op, alpha, beta);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_APPLY_ALPHA_BETA_HPP
