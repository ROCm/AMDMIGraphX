#include <migraphx/simplify_qdq.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/quant_convolution.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/quant_dot.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_set<std::string> get_quantizable_op_names()
{
    static std::unordered_set<std::string> s = {"convolution", "dot"};
    return s;
}

struct match_find_add_bias
{
    auto matcher() const
    {
        return match::name("dequantizelinear")(match::arg(0)(
            match::name("quantizelinear")(
                match::arg(0)(match::name("add")(
                                  match::arg(0)(match::name("convolution").bind("convolution")))
                                  .bind("add")))
                .bind("quantizelinear")));
    }

    void apply(module& m, match::matcher_result r) const
    {
        auto dq   = r.result;
        auto q    = r.instructions["quantizelinear"];
        auto add  = r.instructions["add"];
        auto conv = r.instructions["convolution"];

        auto q_args = q->inputs();
        for(auto&& arg : q_args)
        {
            if(arg->name() == "multibroadcast" or arg->name() == "broadcast")
            {
                arg = m.insert_instruction(add, arg->get_operator(), arg->inputs());
            }
        }
        m.replace_instruction(q, q->get_operator(), q_args);
        q_args.front() = conv;
        auto new_q     = m.insert_instruction(add, migraphx::make_op("quantizelinear"), q_args);
        auto dq_args   = dq->inputs();
        q_args.front() = dq_args.front();
        m.replace_instruction(dq, dq->get_operator(), q_args);
        q_args.front()   = new_q;
        auto new_dq      = m.insert_instruction(add, migraphx::make_op("dequantizelinear"), q_args);
        auto add_args    = add->inputs();
        add_args.front() = new_dq;
        m.replace_instruction(add, migraphx::make_op("add"), add_args);
    }
};

struct match_find_quantizable_ops
{
    auto matcher() const
    {
        return match::name("dequantizelinear")(match::arg(0)(match::name("quantizelinear")(
            match::arg(0)(match::name(get_quantizable_op_names()).bind("qop")))));
    }

    void apply(module& m, match::matcher_result r) const
    {
        auto dq   = r.result;
        auto ins  = r.instructions["qop"];
        auto args = ins->inputs();

        if(std::all_of(args.begin(), args.end(), [&](auto& arg) {
               auto name       = arg->name();
               arg             = arg->inputs().front();
               auto input_type = arg->get_shape().type();
               return name == "dequantizelinear" and input_type == migraphx::shape::int8_type;
           }))
        {
            if(ins->name() == "convolution")
            {
                auto conv_op = any_cast<op::convolution>(ins->get_operator());
                m.replace_instruction(ins,
                                      migraphx::make_op("quant_convolution",
                                                        {{"padding", conv_op.padding},
                                                         {"stride", conv_op.stride},
                                                         {"dilation", conv_op.dilation},
                                                         {"padding_mode", conv_op.padding_mode},
                                                         {"group", conv_op.group}}),
                                      args);
            }
            else if(ins->name() == "dot")
            {
                auto dot_op = any_cast<op::dot>(ins->get_operator());
                m.replace_instruction(
                    ins,
                    migraphx::make_op("quant_dot",
                                      {{"alpha", static_cast<int32_t>(dot_op.alpha)},
                                       {"beta", static_cast<int32_t>(dot_op.beta)}}),
                    args);
            }
            else
            {
                MIGRAPHX_THROW(ins->name() +
                               " does not currently have a quantized implementation.");
            }
            auto dq_args    = dq->inputs();
            dq_args.front() = ins;
            m.replace_instruction(dq, dq->get_operator(), dq_args);
        }
    }
};

void remove_qdq_pairs(module& m)
{
    for(auto ins : iterator_for(m))
    {
        bool replace_op = false;
        auto args       = ins->inputs();
        for(auto&& arg : args)
        {
            if(arg->name() == "dequantizelinear")
            {
                auto q = arg->inputs().front();
                if(q->name() == "quantizelinear")
                {
                    arg        = q->inputs().front();
                    replace_op = true;
                }
            }
        }
        if(replace_op)
        {
            m.replace_instruction(ins, ins->get_operator(), args);
        }
    }
}

void simplify_qdq::apply(module& m) const
{
    match::find_matches(m, match_find_add_bias{});
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
    match::find_matches(m, match_find_quantizable_ops{});
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
    remove_qdq_pairs(m);
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
