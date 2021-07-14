#include <migraphx/remove_qdq_pairs.hpp>
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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_map<std::string, std::string>& quant_ops_map()
{
    static std::unordered_map<std::string, std::string> m = {
        {"convolution", "quant_convolution"},
        {"dot", "quant_dot"}
    };
    return m; 
}

std::unordered_set<std::string> get_quantized_operators()
{
    std::unordered_set<std::string> result;
    std::transform(quant_ops_map().begin(), quant_ops_map().end(), std::inserter(result, result.begin()), [&](auto&& p) {
        return p.first;
    });
    return result;
}

struct match_find_add_bias
{
    auto matcher() const { return match::name("dequantizelinear")(
                                    match::arg(0)(match::name("quantizelinear")(
                                        match::arg(0)(match::name("add")(
                                            match::arg(0)(match::name("convolution").bind("convolution"))
                                        ).bind("add"))
                                    ).bind("quantizelinear"))); }
    void apply(module& mod, match::matcher_result r) const
    {
        auto dq = r.result;
        auto q = r.instructions["quantizelinear"];
        auto add = r.instructions["add"];
        auto conv = r.instructions["convolution"];

        std::cout << "dq  : " << dq->name() << std::endl
                  << "q   : " << q->name() << std::endl
                  << "add : " << add->name() << std::endl
                  << "conv: " << conv->name() << std::endl;

        auto q_args = q->inputs();
        for (auto&& arg : q_args)
        {
            if (arg->name() == "multibroadcast" or arg->name() == "broadcast")
            {
                arg = mod.insert_instruction(add, arg->get_operator(), arg->inputs());
            }
        }
        mod.replace_instruction(q, q->get_operator(), q_args);
        q_args.front() = conv;
        auto new_q = mod.insert_instruction(add, migraphx::make_op("quantizelinear"), q_args);
        auto dq_args = dq->inputs();
        q_args.front() = dq_args.front();
        /*
        for (auto&& arg : dq_args)
        {
            if (arg->name() == "multibroadcast" or arg->name() == "broadcast")
            {
                arg = mod.insert_instruction(add, arg->get_operator(), arg->inputs());
            }
        }*/
        mod.replace_instruction(dq, dq->get_operator(), q_args);
        q_args.front() = new_q;
        auto new_dq = mod.insert_instruction(add, migraphx::make_op("dequantizelinear"), q_args);
        auto add_args = add->inputs();
        add_args.front() = new_dq;
        mod.replace_instruction(add, migraphx::make_op("add"), add_args);
        
        return;
    }
};

struct match_find_dqopq
{
    auto matcher() const { return match::name("dequantizelinear")(
                                    match::arg(0)(match::name("quantizelinear")(
                                        match::arg(0)(match::name(get_quantized_operators()).bind("qop"))
                                    ).bind("quantizelinear"))); }
    void apply(module& mod, match::matcher_result r) const
    {
        auto dq = r.result;
        auto q = r.instructions["quantizelinear"];
        auto ins = r.instructions["qop"];
        auto args = ins->inputs();
        bool replace = false;
        bool mismatch = false;

        std::cout << "q  : " << q->name() << std::endl
                  << "qop: " << ins->name() << std::endl
                  << "dq : " << dq->name() << std::endl;
        
        for (auto&& arg : args)
        {
            if (arg->name() == "dequantizelinear")
            {
                auto q1 = arg->inputs().front();
                if (q1->get_shape().type() == migraphx::shape::int8_type)
                {
                    arg = q1;
                    replace = true;
                }
                else 
                {
                    mismatch = true;
                }
            }
        }

        if (replace and not mismatch)
        {
            std::cout << ins->get_operator() << ", " << ins->get_shape() << std::endl <<  " replaced with " << std::endl; 
            if (ins->name() == "convolution")
            {
                auto conv_op       = any_cast<op::convolution>(ins->get_operator());
                auto padding       = conv_op.padding;
                auto stride        = conv_op.stride;
                auto dilation      = conv_op.dilation;
                auto padding_mode  = conv_op.padding_mode;
                auto group         = conv_op.group;
                mod.replace_instruction(ins, op::quant_convolution{padding, stride, dilation, padding_mode, group}, args);
            }
            else 
            {
                mod.replace_instruction(ins, migraphx::make_op(quant_ops_map().at(ins->name())), args);
            }
            std::cout << ins->get_operator() << ", " << ins->get_shape() << std::endl;
            auto dq_args = dq->inputs();
            dq_args.front() = ins; 
            mod.replace_instruction(dq, dq->get_operator(), dq_args);
        }
        return;
    }
};
/*
struct match_find_qdq
{
    auto matcher() const { return ;//match::any()(match::either_arg(0, 1)(match::name("dequantizelinear"), match::name("dequantizelinear"))); }
    void apply(module& mod, match::matcher_result r) const
    {
        auto res = r.result;
        std::cout << "Match: " << res->name() << std::endl;
        return;
    }
};
*/
void remove_qdq_pairs::apply(module& m) const
{
    //match::find_matches(m, match_find_qdq{});
    //return;
    match::find_matches(m, match_find_add_bias{});
    std::cout << "Add bias pass: " << std::endl;
    m.debug_print();
    match::find_matches(m, match_find_dqopq{});
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
    std::cout << "QOp pass: " << std::endl;
    m.debug_print();
    
    for (auto ins : iterator_for(m))
    {
        bool replace_op = false;
        auto args = ins->inputs();
        for (auto&& arg : args)
        {
            if (arg->name() == "dequantizelinear") 
            {
                auto q = arg->inputs().front();
                std::cout << "q ins: " << q->name() << std::endl;
                if (q->name() == "quantizelinear")
                {
                    arg = q->inputs().front();
                    replace_op = true;
                }
            }
        }
        if (replace_op)
        {
            m.replace_instruction(ins, ins->get_operator(), args);
        }
    }
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});

    return;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
