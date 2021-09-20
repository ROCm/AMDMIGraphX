#ifndef MIGRAPHX_GUARD_MIGRAPHX_COMMON_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_COMMON_HPP

#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
struct operation;

std::vector<std::size_t> compute_broadcasted_lens(std::vector<std::size_t> s0,
                                                  std::vector<std::size_t> s1);
shape common_shape(const std::vector<shape>& shapes);

instruction_ref insert_common_op(module& m,
                                 instruction_ref ins,
                                 const operation& op,
                                 std::vector<instruction_ref> inputs);
instruction_ref add_common_op(module& m, const operation& op, std::vector<instruction_ref> inputs);


template<typename T = float>
instruction_ref insert_dot_apply_alpha_beta(module& m,
                                            instruction_ref pos,
                                            const std::vector<instruction_ref>& args,
                                            std::string op_name,
                                            T alpha = 1.0f,
                                            T beta = 0.0f)
{
    auto l1       = args[0];
    auto l2       = args[1];
    auto dot_type = l1->get_shape().type();
    if(!float_equal(static_cast<float>(alpha), 1.0f))
    {
        auto alpha_literal = m.add_literal(alpha);
        l1 = insert_common_op(m, pos, migraphx::make_op("mul"), {alpha_literal, l1});
        if(l1->get_shape().type() != dot_type)
        {
            l1 = m.insert_instruction(pos, make_op("convert", {{"target_type", dot_type}}), l1);
        }
    }
    auto dot_res = m.insert_instruction(pos, migraphx::make_op(op_name), l1, l2);
    if(args.size() == 3)
    {
        if(not float_equal(static_cast<float>(beta), 0.0f) && args[2]->get_shape().elements() > 0)
        {
            auto out_lens   = l1->get_shape().lens();
            out_lens.back() = l2->get_shape().lens().back();
            auto l3         = args[2];
            auto l3_lens    = l3->get_shape().lens();
            dot_type = l3->get_shape().type();
            if(!std::equal(out_lens.begin(), out_lens.end(), l3_lens.begin(), l3_lens.end()))
            {
                l3 = m.insert_instruction(
                    pos, migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), args[2]);
            }
            auto beta_literal = m.add_literal(beta);
            auto beta_l3 = insert_common_op(m, pos, migraphx::make_op("mul"), {l3, beta_literal});
            if(beta_l3->get_shape().type() != dot_type)
            {
                beta_l3 = m.insert_instruction(
                    pos, migraphx::make_op("convert", {{"target_type", dot_type}}), beta_l3);
            }
            return m.insert_instruction(pos, migraphx::make_op("add"), dot_res, beta_l3);
        }
    }
    return dot_res;
}

template<typename T = float>
instruction_ref add_dot_apply_alpha_beta(module& m,
                                         const std::vector<instruction_ref>& args,
                                         std::string op_name,
                                         T alpha = 1.0f,
                                         T beta = 0.0f)
{
    return insert_dot_apply_alpha_beta(m, m.end(), args, op_name, alpha, beta);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_COMMON_HPP
