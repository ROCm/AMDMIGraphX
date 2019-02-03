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
    std::unordered_map<instruction_ref, instruction_ref> map_last_output;
    for(auto ins : iterator_for(prog))
    {
        if(ins->name() == "gru")
        {
            // could be 3 to 5 inputs (though onnx::rnn has 6 inputs,
            // the 5th one is undefined and ignored by protobuf. so
            // we need to process up to 5 inputs
            auto args = ins->inputs();

            shape seq_shape         = args[0]->get_shape();
            std::size_t hidden_size = args[2]->get_shape().lens()[2];
            std::size_t batch_size  = seq_shape.lens()[1];
            shape::type_t type      = seq_shape.type();
            migraphx::shape ih_shape{type, {1, batch_size, hidden_size}};
            std::vector<char> data(ih_shape.bytes(), 0);

            auto gru_op                    = any_cast<op::gru>(ins->get_operator());
            op::gru::gru_direction_t dicrt = gru_op.direction;
            if(dicrt == op::gru::bidirectional)
            {
                // w weight matrix
                auto w_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[1]);
                auto w_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[1]);

                // r weight matrix
                auto r_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[2]);
                auto r_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[2]);

                // bias
                instruction_ref bias_forward, bias_reverse;
                bias_forward = bias_reverse = prog.end();
                if(args.size() >= 4 && args[3]->get_operator().name() != "undefined")
                {
                    bias_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[3]);
                    bias_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[3]);
                }

                // intial hidden state
                instruction_ref ih_forward, ih_reverse;
                if(args.size() == 6 && args[5]->get_operator().name() != "undefined")
                {
                    ih_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[5]);
                    ih_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[5]);
                }
                else
                {
                    ih_forward = prog.add_literal(migraphx::literal{ih_shape, data});
                    ih_reverse = prog.add_literal(migraphx::literal{ih_shape, data});
                }

                auto ret_forward = gru_cell(true,
                                            prog,
                                            ins,
                                            args[0],
                                            w_forward,
                                            r_forward,
                                            bias_forward,
                                            ih_forward,
                                            gru_op.linear_before_reset,
                                            gru_op.actv_funcs.at(0),
                                            gru_op.actv_funcs.at(1));

                auto ret_reverse = gru_cell(false,
                                            prog,
                                            ins,
                                            args[0],
                                            w_reverse,
                                            r_reverse,
                                            bias_reverse,
                                            ih_reverse,
                                            gru_op.linear_before_reset,
                                            gru_op.actv_funcs.at(2),
                                            gru_op.actv_funcs.at(3));

                auto concat_output =
                    prog.insert_instruction(ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
                auto last_output = prog.insert_instruction(ins, op::squeeze{{0}}, concat_output);

                // The following logic is to ensure the last instruction rewritten
                // from gru operator is a concat
                instruction_ref hidden_state{};
                if (ret_forward[0] == prog.end())
                {
                    hidden_state = prog.replace_instruction(ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
                }
                else 
                {
                    ret_forward[0] =
                        prog.insert_instruction(ins, op::concat{0}, ret_forward[0], ret_forward[1]);
                    ret_reverse[0] =
                        prog.insert_instruction(ins, op::concat{0}, ret_reverse[1], ret_reverse[0]);
                    hidden_state = prog.replace_instruction(
                        ins, op::concat{1}, {ret_forward[0], ret_reverse[0]});
                } 
                map_last_output[hidden_state] = last_output;
            }
            else
            {
                bool is_forward = (dicrt == op::gru::forward) ? true : false;
                // weight matrix
                auto w = args[1];
                auto r = args[2];

                // bias
                instruction_ref bias = prog.end();
                if(args.size() >= 4 && args[3]->get_operator().name() != "undefined")
                {
                    bias = args[3];
                }

                // intial hidden state
                instruction_ref ih;
                if(args.size() == 6 && args[5]->get_operator().name() != "undefined")
                {
                    ih = args[5];
                }
                else
                {
                    ih = prog.add_literal(migraphx::literal{ih_shape, data});
                }

                auto ret = gru_cell(is_forward,
                                    prog,
                                    ins,
                                    args[0],
                                    w,
                                    r,
                                    bias,
                                    ih,
                                    gru_op.linear_before_reset,
                                    gru_op.actv_funcs.at(0),
                                    gru_op.actv_funcs.at(1));

                auto last_output = prog.insert_instruction(ins, op::squeeze{{0}}, ret[1]);

                instruction_ref hidden_state{};
                if (ret[0] == prog.end())
                {
                    hidden_state = prog.replace_instruction(ins, op::concat{0}, ret[1]);
                }
                else
                {
                    auto concat_arg0 = is_forward ? ret[0] : ret[1];
                    auto concat_arg1 = is_forward ? ret[1] : ret[1];
                    hidden_state = prog.replace_instruction(ins, op::concat{0}, concat_arg0, concat_arg1);
                }
                map_last_output[hidden_state] = last_output;
            }
        }

        // rewrite the gru_last_output operator that right after the gru
        // operator. Intuitively, we can do a slice on its input to get
        // the last output, but it is already existed in the rnn operator,
        // so we can just use it as the output here
        if(ins->name() == "gru_last_output")
        {
            auto inputs = ins->inputs();
            assert(inputs.size() == 1);
            assert(map_last_output.count(inputs[0]) > 0);
            prog.replace_instruction(ins, map_last_output[inputs[0]]);
        }
    }
}

std::vector<instruction_ref> rewrite_gru::gru_cell(bool is_forward,
                                                   program& prog,
                                                   instruction_ref ins,
                                                   instruction_ref input,
                                                   instruction_ref w,
                                                   instruction_ref r,
                                                   instruction_ref bias,
                                                   instruction_ref ih,
                                                   int linear_before_reset,
                                                   operation& actv_func1,
                                                   operation& actv_func2) const
{
    instruction_ref hidden_states = prog.end(), last_output;
    long seq_len = static_cast<long>(input->get_shape().lens()[0]);
    long hs      = static_cast<long>(r->get_shape().lens()[2]);

    migraphx::shape s(input->get_shape().type(),
                      {input->get_shape().lens()[1], static_cast<std::size_t>(hs)});
    std::vector<int> data(s.elements(), 1);
    auto l1 = prog.add_literal(migraphx::literal{s, data});

    // weight matrix
    std::vector<int64_t> perm{1, 0};
    auto sw      = prog.insert_instruction(ins, op::squeeze{{0}}, w);
    auto wz      = prog.insert_instruction(ins, op::slice{{0}, {0}, {hs}}, sw);
    auto tran_wz = prog.insert_instruction(ins, op::transpose{perm}, wz);

    auto wr      = prog.insert_instruction(ins, op::slice{{0}, {hs}, {2 * hs}}, sw);
    auto tran_wr = prog.insert_instruction(ins, op::transpose{perm}, wr);

    auto wh      = prog.insert_instruction(ins, op::slice{{0}, {2 * hs}, {3 * hs}}, sw);
    auto tran_wh = prog.insert_instruction(ins, op::transpose{perm}, wh);

    auto sr      = prog.insert_instruction(ins, op::squeeze{{0}}, r);
    auto rz      = prog.insert_instruction(ins, op::slice{{0}, {0}, {hs}}, sr);
    auto tran_rz = prog.insert_instruction(ins, op::transpose{perm}, rz);

    auto rr      = prog.insert_instruction(ins, op::slice{{0}, {hs}, {2 * hs}}, sr);
    auto tran_rr = prog.insert_instruction(ins, op::transpose{perm}, rr);

    auto rh      = prog.insert_instruction(ins, op::slice{{0}, {2 * hs}, {3 * hs}}, sr);
    auto tran_rh = prog.insert_instruction(ins, op::transpose{perm}, rh);

    // initial states
    auto sih = prog.insert_instruction(ins, op::squeeze{{0}}, ih);

    // bias
    instruction_ref brcst_bz, brcst_br, brcst_wbh, brcst_rbh, brcst_bh;
    if(bias != prog.end())
    {
        auto sbias = prog.insert_instruction(ins, op::squeeze{{0}}, bias);
        auto wbz   = prog.insert_instruction(ins, op::slice{{0}, {0}, {hs}}, sbias);
        auto wbr   = prog.insert_instruction(ins, op::slice{{0}, {hs}, {2 * hs}}, sbias);
        auto wbh   = prog.insert_instruction(ins, op::slice{{0}, {2 * hs}, {3 * hs}}, sbias);
        brcst_wbh  = prog.insert_instruction(ins, op::broadcast{1, sih->get_shape()}, wbh);

        auto rbz  = prog.insert_instruction(ins, op::slice{{0}, {3 * hs}, {4 * hs}}, sbias);
        auto rbr  = prog.insert_instruction(ins, op::slice{{0}, {4 * hs}, {5 * hs}}, sbias);
        auto rbh  = prog.insert_instruction(ins, op::slice{{0}, {5 * hs}, {6 * hs}}, sbias);
        brcst_rbh = prog.insert_instruction(ins, op::broadcast{1, sih->get_shape()}, rbh);

        auto bz  = prog.insert_instruction(ins, op::add{}, wbz, rbz);
        brcst_bz = prog.insert_instruction(ins, op::broadcast{1, sih->get_shape()}, bz);

        auto br  = prog.insert_instruction(ins, op::add{}, wbr, rbr);
        brcst_br = prog.insert_instruction(ins, op::broadcast{1, sih->get_shape()}, br);

        auto bh  = prog.insert_instruction(ins, op::add{}, wbh, rbh);
        brcst_bh = prog.insert_instruction(ins, op::broadcast{1, sih->get_shape()}, bh);
    }

    for(long i = 0; i < seq_len; i++)
    {
        long seq_index = is_forward ? i : (seq_len - 1 - i);
        auto xt = prog.insert_instruction(ins, op::slice{{0}, {seq_index}, {seq_index + 1}}, input);
        xt      = prog.insert_instruction(ins, op::squeeze{{0}}, xt);

        // equation f(xt*(Wz^T) + Ht-1 * (Rz^T) + Wbz + Rbz)
        auto xt_wz = prog.insert_instruction(ins, op::dot{}, xt, tran_wz);
        auto ht_rz = prog.insert_instruction(ins, op::dot{}, sih, tran_rz);
        auto xht_z = prog.insert_instruction(ins, op::add{}, xt_wz, ht_rz);
        if(bias != prog.end())
        {
            xht_z = prog.insert_instruction(ins, op::add{}, xht_z, brcst_bz);
        }
        auto zt = prog.insert_instruction(ins, actv_func1, xht_z);

        // equation f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        auto xt_wr = prog.insert_instruction(ins, op::dot{}, xt, tran_wr);
        auto ht_rr = prog.insert_instruction(ins, op::dot{}, sih, tran_rr);
        auto xht_r = prog.insert_instruction(ins, op::add{}, xt_wr, ht_rr);
        if(bias != prog.end())
        {
            xht_r = prog.insert_instruction(ins, op::add{}, xht_r, brcst_br);
        }
        auto rt = prog.insert_instruction(ins, actv_func1, xht_r);

        instruction_ref xht_h;
        if(linear_before_reset == 0)
        {
            // equation g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
            auto xt_wh  = prog.insert_instruction(ins, op::dot{}, xt, tran_wh);
            auto rt_ht1 = prog.insert_instruction(ins, op::mul{}, rt, sih);
            auto rt_rh  = prog.insert_instruction(ins, op::dot{}, rt_ht1, tran_rh);
            xht_h       = prog.insert_instruction(ins, op::add{}, xt_wh, rt_rh);
            if(bias != prog.end())
            {
                xht_h = prog.insert_instruction(ins, op::add{}, xht_h, brcst_bh);
            }
        }
        else
        {
            // equation ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
            auto xt_wh  = prog.insert_instruction(ins, op::dot{}, xt, tran_wh);
            auto ht1_rh = prog.insert_instruction(ins, op::dot{}, sih, tran_rh);
            if(bias != prog.end())
            {
                ht1_rh = prog.insert_instruction(ins, op::add{}, ht1_rh, brcst_rbh);
            }
            auto rt_rh = prog.insert_instruction(ins, op::mul{}, rt, ht1_rh);
            xht_h      = prog.insert_instruction(ins, op::add{}, xt_wh, rt_rh);
            if(bias != prog.end())
            {
                xht_h = prog.insert_instruction(ins, op::add{}, xht_h, brcst_wbh);
            }
        }
        auto ht = prog.insert_instruction(ins, actv_func2, xht_h);

        // equation Ht = (1 - zt) (.) ht + zt (.) Ht-1
        auto one_minus_zt    = prog.insert_instruction(ins, op::sub{}, l1, zt);
        auto one_minus_zt_ht = prog.insert_instruction(ins, op::mul{}, one_minus_zt, ht);
        auto zt_ht1          = prog.insert_instruction(ins, op::mul{}, zt, sih);
        sih                  = prog.insert_instruction(ins, op::add{}, one_minus_zt_ht, zt_ht1);
        last_output          = prog.insert_instruction(ins, op::unsqueeze{{0, 1}}, sih);

        if (i < seq_len - 1)
        {
            if(is_forward)
            {
                hidden_states = (seq_index == 0)
                                 ? last_output
                                 : prog.insert_instruction(ins, op::concat{0}, hidden_states, last_output);
            }
            else
            {
                hidden_states = (seq_index == seq_len - 1)
                                 ? last_output
                                 : prog.insert_instruction(ins, op::concat{0}, last_output, hidden_states);
            }
        }
    }

    return {hidden_states, last_output};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
