#ifndef MIGRAPHX_GUARD_MIGRAPHX_APPLY_ALPHA_BETA_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_APPLY_ALPHA_BETA_HPP

#include "migraphx/errors.hpp"
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/module.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
template <typename T = float>
instruction_ref insert_apply_alpha_beta(module& m,
                                        instruction_ref pos,
                                        const std::vector<instruction_ref>& args,
                                        const std::string& op_name,
                                        T alpha = 1.0f,
                                        T beta  = 0.0f)
{
    auto a        = args[0];
    auto b        = args[1];
    auto dot_type = a->get_shape().type();
    if(!float_equal(alpha, 1.0f))
    {
        auto alpha_literal = m.add_literal(alpha);
        a                  = insert_common_op(m, pos, migraphx::make_op("mul"), {alpha_literal, a});
        if(a->get_shape().type() != dot_type)
        {
            a = m.insert_instruction(pos, make_op("convert", {{"target_type", dot_type}}), a);
        }
    }
    auto dot_res = m.insert_instruction(pos, migraphx::make_op(op_name), a, b);
    if(args.size() == 3)
    {
        if(not float_equal(beta, 0.0f) && args[2]->get_shape().elements() > 0)
        {
            auto out_lens   = a->get_shape().lens();
            out_lens.back() = b->get_shape().lens().back();
            auto c          = args[2];
            auto c_lens     = c->get_shape().lens();
            dot_type        = c->get_shape().type();
            if(!std::equal(out_lens.begin(), out_lens.end(), c_lens.begin(), c_lens.end()))
            {
                c = m.insert_instruction(
                    pos, migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), args[2]);
            }
            auto beta_literal = m.add_literal(beta);
            auto beta_c = insert_common_op(m, pos, migraphx::make_op("mul"), {c, beta_literal});
            if(beta_c->get_shape().type() != dot_type)
            {
                beta_c = m.insert_instruction(
                    pos, migraphx::make_op("convert", {{"target_type", dot_type}}), beta_c);
            }
            return m.insert_instruction(pos, migraphx::make_op("add"), dot_res, beta_c);
        }
    }
    return dot_res;
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
