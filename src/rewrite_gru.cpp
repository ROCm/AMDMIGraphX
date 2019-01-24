#include <migraphx/rewrite_gru.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/dfor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_gru::apply(program& prog) const
{
    for(auto ins : iterator_for(prog))
    {
        if(ins->name() != "gru")
        {
            continue;
        }

        // could be 3 to 5 inputs (though onnx::rnn has 6 inputs,
        // the 5th one is undefined and ignored by protobuf. so
        // we need to process up to 5 inputs
        auto args = ins->inputs();

        shape seq_shape         = args[0]->get_shape();
        std::size_t hidden_size = args[2]->get_shape().lens()[2];
        std::size_t batchs      = seq_shape.lens()[1];
        shape::type_t type      = seq_shape.type();
        migraphx::shape ih_shape{type, {batchs, hidden_size}};
        std::vector<char> data(ih_shape.bytes(), 0);

        auto gru_op                    = any_cast<op::gru>(ins->get_operator());
        op::gru::gru_direction_t dicrt = gru_op.direction;
        if(dicrt == op::gru::bidirectional)
        {
            // forward weight
            auto uw_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[1]);
            auto w_forward  = prog.insert_instruction(ins, op::squeeze{{0}}, uw_forward);

            auto ur_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[2]);
            auto r_forward  = prog.insert_instruction(ins, op::squeeze{{0}}, ur_forward);

            // reverse weight
            auto uw_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[1]);
            auto w_reverse  = prog.insert_instruction(ins, op::squeeze{{0}}, uw_reverse);

            auto ur_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[2]);
            auto r_reverse  = prog.insert_instruction(ins, op::squeeze{{0}}, ur_reverse);

            // process bias
            instruction_ref bias_forward, bias_reverse;
            bias_forward = bias_reverse = prog.end();
            if(args.size() >= 4)
            {
                // forward bias
                auto uwb_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[3]);
                bias_forward     = prog.insert_instruction(ins, op::squeeze{{0}}, uwb_forward);

                // backward bias
                auto uwb_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[3]);
                bias_reverse     = prog.insert_instruction(ins, op::squeeze{{0}}, uwb_reverse);
            }

            // intial hidden state
            instruction_ref ih_forward, ih_reverse;
            if(args.size() >= 5)
            {
                // forward
                ih_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[4]);
                ih_forward = prog.insert_instruction(ins, op::squeeze{{0}}, ih_forward);

                // reverse
                ih_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[4]);
                ih_reverse = prog.insert_instruction(ins, op::squeeze{{0}}, ih_reverse);
            }
            else
            {
                ih_forward = prog.add_literal(migraphx::literal{ih_shape, data});
                ih_reverse = prog.add_literal(migraphx::literal{ih_shape, data});
            }

            auto ret_forward = gru_oper(true,
                                        prog,
                                        ins,
                                        args[0],
                                        w_forward,
                                        r_forward,
                                        ih_forward,
                                        bias_forward,
                                        gru_op.linear_before_reset,
                                        gru_op.actv_funcs.at(0),
                                        gru_op.actv_funcs.at(1));

            auto ret_reverse = gru_oper(false,
                                        prog,
                                        ins,
                                        args[0],
                                        w_reverse,
                                        r_reverse,
                                        ih_reverse,
                                        bias_reverse,
                                        gru_op.linear_before_reset,
                                        gru_op.actv_funcs.at(2),
                                        gru_op.actv_funcs.at(3));

            auto final_output = prog.insert_instruction(ins, op::concat{0}, ret_forward[1], ret_reverse[1]);

            // add the dimension of num_direction
            ret_forward[0] = prog.insert_instruction(ins, op::unsqueeze{{1}}, ret_forward[0]);
            ret_reverse[0] = prog.insert_instruction(ins, op::unsqueeze{{1}}, ret_reverse[0]);

            // concat the forward and reverse output
            auto replaced_arg = prog.replace_instruction(ins, op::concat{1}, {ret_forward[0], ret_reverse[0]});
            replaced_arg->add_output(final_output);
        }
        else
        {
            bool is_forward = (dicrt == op::gru::forward) ? true : false;
            // weight matrix
            auto w = prog.insert_instruction(ins, op::squeeze{{0}}, args[1]);
            auto r = prog.insert_instruction(ins, op::squeeze{{0}}, args[2]);

            // bias
            instruction_ref bias = prog.end();
            if(args.size() >= 4)
            {
                bias = prog.insert_instruction(ins, op::squeeze{{0}}, args[3]);
            }

            // intial hidden state
            instruction_ref ih;
            if(args.size() >= 5)
            {
                ih = prog.insert_instruction(ins, op::squeeze{{0}}, args[4]);
            }
            else
            {
                ih = prog.add_literal(migraphx::literal{ih_shape, data});
            }

            auto ret = gru_oper(is_forward,
                                prog,
                                ins,
                                args[0],
                                w,
                                r,
                                ih,
                                bias,
                                gru_op.linear_before_reset,
                                gru_op.actv_funcs.at(0),
                                gru_op.actv_funcs.at(1));

            // add the dimension of num_direction
            auto replaced_arg = prog.replace_instruction(ins, op::unsqueeze{{1}}, ret[0]);
            replaced_arg->add_output(ret[1]);
        }
    }
}

std::vector<instruction_ref> rewrite_gru::gru_oper(bool is_forward,
                                                   program& prog,
                                                   instruction_ref ins,
                                                   instruction_ref input,
                                                   instruction_ref w,
                                                   instruction_ref r,
                                                   instruction_ref ih,
                                                   instruction_ref bias,
                                                   int linear_before_reset,
                                                   operation& actv_func1,
                                                   operation& actv_func2) const
{
    instruction_ref hidden_out, final_out;
    long seq_len   = static_cast<long>(input->get_shape().lens()[0]);
    long hs        = static_cast<long>(r->get_shape().lens()[1]);
    long seq_index = is_forward ? 0 : seq_len - 1;

    migraphx::shape s(input->get_shape().type(),
                      {input->get_shape().lens()[1], static_cast<std::size_t>(hs)});
    std::vector<int> data(s.elements(), 1);
    auto l1 = prog.add_literal(migraphx::literal{s, data});

    // weight matrix
    std::vector<int64_t> perm{1, 0};
    auto wz  = prog.insert_instruction(ins, op::slice{{0}, {0}, {hs}}, w);
    auto twz = prog.insert_instruction(ins, op::transpose{perm}, wz);
    auto wr  = prog.insert_instruction(ins, op::slice{{0}, {hs}, {2 * hs}}, w);
    auto twr = prog.insert_instruction(ins, op::transpose{perm}, wr);
    auto wh  = prog.insert_instruction(ins, op::slice{{0}, {2 * hs}, {3 * hs}}, w);
    auto twh = prog.insert_instruction(ins, op::transpose{perm}, wh);

    auto rz  = prog.insert_instruction(ins, op::slice{{0}, {0}, {hs}}, r);
    auto trz = prog.insert_instruction(ins, op::transpose{perm}, rz);
    auto rr  = prog.insert_instruction(ins, op::slice{{0}, {hs}, {2 * hs}}, r);
    auto trr = prog.insert_instruction(ins, op::transpose{perm}, rr);
    auto rh  = prog.insert_instruction(ins, op::slice{{0}, {2 * hs}, {3 * hs}}, r);
    auto trh = prog.insert_instruction(ins, op::transpose{perm}, rh);

    // bias
    instruction_ref br_bz, br_br, br_wbh, br_rbh, br_bh;
    if(bias != prog.end())
    {
        auto wbz = prog.insert_instruction(ins, op::slice{{0}, {0}, {hs}}, bias);
        auto wbr = prog.insert_instruction(ins, op::slice{{0}, {hs}, {2 * hs}}, bias);
        auto wbh = prog.insert_instruction(ins, op::slice{{0}, {2 * hs}, {3 * hs}}, bias);
        br_wbh   = prog.insert_instruction(ins, op::broadcast{1, ih->get_shape()}, wbh);

        auto rbz = prog.insert_instruction(ins, op::slice{{0}, {3 * hs}, {4 * hs}}, bias);
        auto rbr = prog.insert_instruction(ins, op::slice{{0}, {4 * hs}, {5 * hs}}, bias);
        auto rbh = prog.insert_instruction(ins, op::slice{{0}, {5 * hs}, {6 * hs}}, bias);
        br_rbh   = prog.insert_instruction(ins, op::broadcast{1, ih->get_shape()}, rbh);

        auto bz = prog.insert_instruction(ins, op::add{}, wbz, rbz);
        br_bz   = prog.insert_instruction(ins, op::broadcast{1, ih->get_shape()}, bz);
        auto br = prog.insert_instruction(ins, op::add{}, wbr, rbr);
        br_br   = prog.insert_instruction(ins, op::broadcast{1, ih->get_shape()}, br);
        auto bh = prog.insert_instruction(ins, op::add{}, wbh, rbh);
        br_bh   = prog.insert_instruction(ins, op::broadcast{1, ih->get_shape()}, bh);
    }

    for(long i = 0; i < seq_len; i++)
    {
        auto xt = prog.insert_instruction(ins, op::slice{{0}, {seq_index}, {seq_index + 1}}, input);
        xt      = prog.insert_instruction(ins, op::squeeze{{0}}, xt);
        // equation f(xt*(Wz^T) + Ht-1 * (Rz^T) + Wbz + Rbz)
        auto xwzt    = prog.insert_instruction(ins, op::dot{}, xt, twz);
        auto hrzt    = prog.insert_instruction(ins, op::dot{}, ih, trz);
        auto xwhr_zt = prog.insert_instruction(ins, op::add{}, xwzt, hrzt);
        if(bias != prog.end())
        {
            xwhr_zt = prog.insert_instruction(ins, op::add{}, xwhr_zt, br_bz);
        }
        auto zt = prog.insert_instruction(ins, actv_func1, xwhr_zt);

        // equation f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        auto xwrt    = prog.insert_instruction(ins, op::dot{}, xt, twr);
        auto hrrt    = prog.insert_instruction(ins, op::dot{}, ih, trr);
        auto xwhr_rt = prog.insert_instruction(ins, op::add{}, xwrt, hrrt);
        if(bias != prog.end())
        {
            xwhr_rt = prog.insert_instruction(ins, op::add{}, xwhr_rt, br_br);
        }
        auto rt = prog.insert_instruction(ins, actv_func1, xwhr_rt);

        instruction_ref xwhh_rt;
        if(linear_before_reset == 0)
        {
            // equation g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
            auto xwht  = prog.insert_instruction(ins, op::dot{}, xt, twh);
            auto rt_ht = prog.insert_instruction(ins, op::mul{}, rt, ih);
            auto rt_rh = prog.insert_instruction(ins, op::dot{}, rt_ht, trh);
            xwhh_rt    = prog.insert_instruction(ins, op::add{}, xwht, rt_rh);
            if(bias != prog.end())
            {
                xwhh_rt = prog.insert_instruction(ins, op::add{}, xwhh_rt, br_bh);
            }
        }
        else
        {
            // equation ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
            auto xwht   = prog.insert_instruction(ins, op::dot{}, xt, twh);
            auto ih_rht = prog.insert_instruction(ins, op::dot{}, ih, trh);
            if(bias != prog.end())
            {
                ih_rht = prog.insert_instruction(ins, op::add{}, ih_rht, br_rbh);
            }
            auto rt_rh = prog.insert_instruction(ins, op::mul{}, rt, ih_rht);
            xwhh_rt    = prog.insert_instruction(ins, op::add{}, xwht, rt_rh);
            if(bias != prog.end())
            {
                xwhh_rt = prog.insert_instruction(ins, op::add{}, xwhh_rt, br_wbh);
            }
        }
        auto ht = prog.insert_instruction(ins, actv_func2, xwhh_rt);

        // equation Ht = (1 - zt) (.) ht + zt (.) Ht-1
        auto z1t   = prog.insert_instruction(ins, op::sub{}, l1, zt);
        auto z1tht = prog.insert_instruction(ins, op::mul{}, z1t, ht);
        auto ztht1 = prog.insert_instruction(ins, op::mul{}, zt, ih);
        ih         = prog.insert_instruction(ins, op::add{}, z1tht, ztht1);
        final_out  = prog.insert_instruction(ins, op::unsqueeze{{0}}, ih);

        if(is_forward)
        {
            hidden_out = (seq_index == 0)
                             ? final_out
                             : prog.insert_instruction(ins, op::concat{0}, hidden_out, final_out);
        }
        else
        {
            hidden_out = (seq_index == seq_len - 1)
                             ? final_out
                             : prog.insert_instruction(ins, op::concat{0}, final_out, hidden_out);
        }
        seq_index = is_forward ? (seq_index + 1) : (seq_index - 1);
    }

    std::vector<instruction_ref> out_args;
    out_args.push_back(hidden_out);
    out_args.push_back(final_out);

    return out_args;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
