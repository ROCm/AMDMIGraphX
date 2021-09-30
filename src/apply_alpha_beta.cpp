#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/apply_alpha_beta.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

instruction_ref insert_apply_alpha_beta(module& m,
                                        instruction_ref pos,
                                        const std::vector<instruction_ref>& args,
                                        const std::string& op_name,
                                        const literal& alpha,
                                        const literal& beta)
{
    auto a          = args[0];
    auto b          = args[1];
    auto input_type = a->get_shape().type();
    if(!float_equal(alpha.at<float>(0), 1.0))
    {
        auto alpha_literal = m.add_literal(alpha);
        a                  = insert_common_op(m, pos, migraphx::make_op("mul"), {alpha_literal, a});
        if(a->get_shape().type() != input_type)
        {
            a = m.insert_instruction(pos, make_op("convert", {{"target_type", input_type}}), a);
        }
    }
    auto dot_res = m.insert_instruction(pos, migraphx::make_op(op_name), a, b);
    if(args.size() == 3)
    {
        if(not float_equal(beta.at<float>(0), 0.0) && args[2]->get_shape().elements() > 0)
        {
            auto out_lens   = a->get_shape().lens();
            out_lens.back() = b->get_shape().lens().back();
            auto c          = args[2];
            auto c_lens     = c->get_shape().lens();
            input_type      = c->get_shape().type();
            if(out_lens != c_lens)
            {
                c = m.insert_instruction(
                    pos, migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), args[2]);
            }
            auto beta_literal = m.add_literal(beta);
            auto beta_c = insert_common_op(m, pos, migraphx::make_op("mul"), {c, beta_literal});
            if(beta_c->get_shape().type() != input_type)
            {
                beta_c = m.insert_instruction(
                    pos, migraphx::make_op("convert", {{"target_type", input_type}}), beta_c);
            }
            return m.insert_instruction(pos, migraphx::make_op("add"), dot_res, beta_c);
        }
    }
    return dot_res;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
