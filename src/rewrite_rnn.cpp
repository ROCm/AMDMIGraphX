#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/dfor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_rnn::apply(program& prog) const
{
    for(auto ins : iterator_for(prog))
    {
        if(ins->name() == "rnn")
        {
            apply_vanilla_rnn(prog, ins);
        }

        if(ins->name() == "gru")
        {
            apply_gru(prog, ins);
        }
    }
}

void rewrite_rnn::apply_vanilla_rnn(program& prog, instruction_ref ins) const
{
    assert(ins->name() == "rnn");
    // could be 3 to 6 inputs, but the parse_rnn function will
    // append undefined operators to make 6 arguments when parsing
    // an onnx file. Another case is user can have num of arguments
    // when writing their program.
    auto args = ins->inputs();

    shape seq_shape         = args[0]->get_shape();
    std::size_t hidden_size = args[1]->get_shape().lens()[1];
    std::size_t batch_size  = seq_shape.lens()[1];
    shape::type_t type      = seq_shape.type();
    migraphx::shape ih_shape{type, {1, batch_size, hidden_size}};
    std::vector<float> data(ih_shape.elements(), 0);

    auto actv_funcs                = vanilla_rnn_actv_funcs(ins);
    auto rnn_op                    = any_cast<op::rnn>(ins->get_operator());
    op::rnn::rnn_direction_t dicrt = rnn_op.direction;
    instruction_ref last_output{};
    if(dicrt == op::rnn::bidirectional)
    {
        // input weight matrix
        auto w_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[1]);
        auto w_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[1]);

        // hidden state weight matrix
        auto r_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[2]);
        auto r_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[2]);

        // process bias
        instruction_ref bias_forward = prog.end();
        instruction_ref bias_reverse = prog.end();
        if(args.size() >= 4 && args[3]->name() != "undefined")
        {
            bias_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[3]);
            bias_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[3]);
        }

        // process intial hidden state, it could be the 6th argument
        // or the 5th one (if the sequence len argument is ignored)
        instruction_ref ih_forward{};
        instruction_ref ih_reverse{};
        if(args.size() == 6 && args[5]->name() != "undefined")
        {
            ih_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[5]);
            ih_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[5]);
        }
        else
        {
            ih_forward = prog.add_literal(migraphx::literal{ih_shape, data});
            ih_reverse = prog.add_literal(migraphx::literal{ih_shape, data});
        }

        auto ret_forward = vanilla_rnn_cell(true,
                                            prog,
                                            ins,
                                            args[0],
                                            w_forward,
                                            r_forward,
                                            bias_forward,
                                            ih_forward,
                                            actv_funcs.at(0));
        auto ret_reverse = vanilla_rnn_cell(false,
                                            prog,
                                            ins,
                                            args[0],
                                            w_reverse,
                                            r_reverse,
                                            bias_reverse,
                                            ih_reverse,
                                            actv_funcs.at(1));

        auto concat_output =
            prog.insert_instruction(ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
        last_output = prog.insert_instruction(ins, op::squeeze{{0}}, concat_output);

        // The following logic is to ensure the last instruction rewritten from
        // rnn operator is a concat instruction
        // sequence len is 1
        instruction_ref hidden_output{};
        if(ret_forward[0] == prog.end())
        {
            hidden_output =
                prog.replace_instruction(ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
        }
        else
        {
            ret_forward[0] =
                prog.insert_instruction(ins, op::concat{0}, ret_forward[0], ret_forward[1]);
            ret_reverse[0] =
                prog.insert_instruction(ins, op::concat{0}, ret_reverse[1], ret_reverse[0]);
            hidden_output =
                prog.replace_instruction(ins, op::concat{1}, {ret_forward[0], ret_reverse[0]});
        }
    }
    else
    {
        bool is_forward = (dicrt == op::rnn::forward);
        // input weight matrix
        auto w = args[1];

        // hidden state weight matrix
        auto r = args[2];

        // process bias and initial hidden state
        instruction_ref bias = prog.end();
        if(args.size() >= 4 && args[3]->name() != "undefined")
        {
            bias = args[3];
        }

        // process intial hidden state
        instruction_ref ih;
        if(args.size() == 6 && args[5]->name() != "undefined")
        {
            ih = args[5];
        }
        else
        {
            ih = prog.add_literal(migraphx::literal{ih_shape, data});
        }

        auto ret =
            vanilla_rnn_cell(is_forward, prog, ins, args[0], w, r, bias, ih, actv_funcs.at(0));
        last_output = prog.insert_instruction(ins, op::squeeze{{0}}, ret[1]);

        // following logic is to ensure the last instruction is a
        // concat instruction
        // sequence len is 1
        instruction_ref hidden_output{};
        if(ret[0] == prog.end())
        {
            hidden_output = prog.replace_instruction(ins, op::concat{0}, ret[1]);
        }
        else
        {
            auto concat_arg0 = is_forward ? ret[0] : ret[1];
            auto concat_arg1 = is_forward ? ret[1] : ret[0];
            hidden_output = prog.replace_instruction(ins, op::concat{0}, concat_arg0, concat_arg1);
        }
    }

    // search its output to find if there are rnn_last_output operator
    // while loop to handle case of multiple rnn_last_output operators
    auto last_output_it = ins->outputs().begin();
    while(last_output_it != ins->outputs().end())
    {
        last_output_it = std::find_if(last_output_it, ins->outputs().end(), [](auto i) {
            return i->name() == "rnn_last_output";
        });

        if(last_output_it != ins->outputs().end())
        {
            prog.replace_instruction(*last_output_it, last_output);
            last_output_it++;
        }
    }
}

std::vector<instruction_ref> rewrite_rnn::vanilla_rnn_cell(bool is_forward,
                                                           program& prog,
                                                           instruction_ref ins,
                                                           instruction_ref input,
                                                           instruction_ref w,
                                                           instruction_ref r,
                                                           instruction_ref bias,
                                                           instruction_ref ih,
                                                           operation& actv_func) const
{
    // squeeze and transpose w
    std::vector<int64_t> perm{1, 0};
    auto sw      = prog.insert_instruction(ins, op::squeeze{{0}}, w);
    auto tran_sw = prog.insert_instruction(ins, op::transpose{perm}, sw);

    // squeeze and transpose r
    auto sr      = prog.insert_instruction(ins, op::squeeze{{0}}, r);
    auto tran_sr = prog.insert_instruction(ins, op::transpose{perm}, sr);

    // initial hidden state
    auto sih = prog.insert_instruction(ins, op::squeeze{{0}}, ih);

    // bias
    if(bias != prog.end())
    {
        long hs    = r->get_shape().lens()[2];
        auto sbias = prog.insert_instruction(ins, op::squeeze{{0}}, bias);
        auto wb    = prog.insert_instruction(ins, op::slice{{0}, {0}, {hs}}, sbias);
        auto rb    = prog.insert_instruction(ins, op::slice{{0}, {hs}, {2 * hs}}, sbias);
        auto b     = prog.insert_instruction(ins, op::add{}, wb, rb);
        bias       = prog.insert_instruction(ins, op::broadcast{1, sih->get_shape()}, b);
    }

    instruction_ref hidden_out = prog.end();
    instruction_ref last_out{};
    last_out            = prog.insert_instruction(ins, op::unsqueeze{{0, 1}}, sih);
    std::size_t seq_len = input->get_shape().lens()[0];
    for(std::size_t i = 0; i < seq_len; i++)
    {
        long seq_index = is_forward ? i : (seq_len - 1 - i);
        auto xt = prog.insert_instruction(ins, op::slice{{0}, {seq_index}, {seq_index + 1}}, input);
        xt      = prog.insert_instruction(ins, op::squeeze{{0}}, xt);
        auto xt_wi = prog.insert_instruction(ins, op::dot{}, xt, tran_sw);
        auto ht_ri = prog.insert_instruction(ins, op::dot{}, sih, tran_sr);
        auto xt_ht = prog.insert_instruction(ins, op::add{}, xt_wi, ht_ri);
        instruction_ref ht;
        if(bias != prog.end())
        {
            ht = prog.insert_instruction(ins, op::add{}, xt_ht, bias);
        }
        else
        {
            ht = xt_ht;
        }

        // apply activation function
        ht  = prog.insert_instruction(ins, actv_func, ht);
        sih = ht;

        // add the dimensions of sequence length (axis 0 for sequence length,
        // axis 1 for num_directions
        last_out = prog.insert_instruction(ins, op::unsqueeze{{0, 1}}, ht);

        // concatenation for the last last_out is performed in the apply()
        // function to ensure the last instruction is concat, then we have
        // output inserted
        if(i < seq_len - 1)
        {
            if(is_forward)
            {
                hidden_out =
                    (seq_index == 0)
                        ? last_out
                        : prog.insert_instruction(ins, op::concat{0}, hidden_out, last_out);
            }
            else
            {
                hidden_out =
                    (seq_index == seq_len - 1)
                        ? last_out
                        : prog.insert_instruction(ins, op::concat{0}, last_out, hidden_out);
            }
        }
    }

    return {hidden_out, last_out};
}

std::vector<operation> rewrite_rnn::vanilla_rnn_actv_funcs(instruction_ref ins) const
{
    auto rnn_op = any_cast<op::rnn>(ins->get_operator());
    // could be 3 to 6 inputs, but the parse_gru function will
    // append undefined operators to make 6 arguments when parsing
    // an onnx file. Another case is user can have any num of arguments
    // when writing their program.
    if(rnn_op.direction == op::rnn::bidirectional)
    {
        if(rnn_op.actv_funcs.empty())
        {
            // default is tanh
            return {op::tanh{}, op::tanh{}};
        }
        else if(rnn_op.actv_funcs.size() == 1)
        {
            return {rnn_op.actv_funcs.at(0), rnn_op.actv_funcs.at(0)};
        }
        else
        {
            return rnn_op.actv_funcs;
        }
    }
    else
    {
        if(rnn_op.actv_funcs.empty())
        {
            // default is tanh
            return {op::tanh{}};
        }
        else
        {
            return rnn_op.actv_funcs;
        }
    }
}

void rewrite_rnn::apply_gru(program& prog, instruction_ref ins) const
{
    assert(ins->name() == "gru");
    const auto actv_funcs = gru_actv_funcs(ins);
    // could be 3 to 5 inputs (though onnx::rnn has 6 inputs,
    // the 5th one is undefined and ignored by protobuf. so
    // we need to process up to 5 inputs
    auto args = ins->inputs();

    shape seq_shape         = args[0]->get_shape();
    std::size_t hidden_size = args[2]->get_shape().lens()[2];
    std::size_t batch_size  = seq_shape.lens()[1];
    shape::type_t type      = seq_shape.type();
    migraphx::shape ih_shape{type, {1, batch_size, hidden_size}};
    std::vector<float> data(ih_shape.elements(), 0.0);

    auto gru_op                    = any_cast<op::gru>(ins->get_operator());
    op::gru::gru_direction_t dicrt = gru_op.direction;
    instruction_ref last_output{};
    if(dicrt == op::gru::bidirectional)
    {
        // w weight matrix
        auto w_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[1]);
        auto w_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[1]);

        // r weight matrix
        auto r_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[2]);
        auto r_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[2]);

        // bias
        instruction_ref bias_forward = prog.end();
        instruction_ref bias_reverse = prog.end();
        if(args.size() >= 4 && args[3]->name() != "undefined")
        {
            bias_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[3]);
            bias_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[3]);
        }

        // intial hidden state
        instruction_ref ih_forward{};
        instruction_ref ih_reverse{};
        if(args.size() == 6 && args[5]->name() != "undefined")
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
                                    {args[0], w_forward, r_forward, bias_forward, ih_forward},
                                    gru_op.linear_before_reset,
                                    actv_funcs.at(0),
                                    actv_funcs.at(1));

        auto ret_reverse = gru_cell(false,
                                    prog,
                                    ins,
                                    {args[0], w_reverse, r_reverse, bias_reverse, ih_reverse},
                                    gru_op.linear_before_reset,
                                    actv_funcs.at(2),
                                    actv_funcs.at(3));

        auto concat_output =
            prog.insert_instruction(ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
        last_output = prog.insert_instruction(ins, op::squeeze{{0}}, concat_output);

        // The following logic is to ensure the last instruction rewritten
        // from gru operator is a concat
        instruction_ref hidden_state{};
        if(ret_forward[0] == prog.end())
        {
            hidden_state =
                prog.replace_instruction(ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
        }
        else
        {
            ret_forward[0] =
                prog.insert_instruction(ins, op::concat{0}, ret_forward[0], ret_forward[1]);
            ret_reverse[0] =
                prog.insert_instruction(ins, op::concat{0}, ret_reverse[1], ret_reverse[0]);
            hidden_state =
                prog.replace_instruction(ins, op::concat{1}, {ret_forward[0], ret_reverse[0]});
        }
    }
    else
    {
        bool is_forward = (dicrt == op::gru::forward);
        // weight matrix
        auto w = args[1];
        auto r = args[2];

        // bias
        instruction_ref bias = prog.end();
        if(args.size() >= 4 && args[3]->name() != "undefined")
        {
            bias = args[3];
        }

        // intial hidden state
        instruction_ref ih{};
        if(args.size() == 6 && args[5]->name() != "undefined")
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
                            {args[0], w, r, bias, ih},
                            gru_op.linear_before_reset,
                            actv_funcs.at(0),
                            actv_funcs.at(1));

        last_output = prog.insert_instruction(ins, op::squeeze{{0}}, ret[1]);

        instruction_ref hidden_state{};
        if(ret[0] == prog.end())
        {
            hidden_state = prog.replace_instruction(ins, op::concat{0}, ret[1]);
        }
        else
        {
            auto concat_arg0 = is_forward ? ret[0] : ret[1];
            auto concat_arg1 = is_forward ? ret[1] : ret[0];
            hidden_state = prog.replace_instruction(ins, op::concat{0}, concat_arg0, concat_arg1);
        }
    }

    // replace the corresponding gru_last_output instruction
    // with the last_output, if gru_last_output exists
    // while loop to handle case of multiple gru_last_output operators
    auto last_output_it = ins->outputs().begin();
    while(last_output_it != ins->outputs().end())
    {
        last_output_it = std::find_if(last_output_it, ins->outputs().end(), [](auto i) {
            return i->name() == "gru_last_output";
        });

        if(last_output_it != ins->outputs().end())
        {
            prog.replace_instruction(*last_output_it, last_output);
            last_output_it++;
        }
    }
}

std::vector<instruction_ref> rewrite_rnn::gru_cell(bool is_forward,
                                                   program& prog,
                                                   instruction_ref ins,
                                                   std::vector<instruction_ref> inputs,
                                                   int linear_before_reset,
                                                   const operation& actv_func1,
                                                   const operation& actv_func2) const
{
    assert(inputs.size() == 5);
    auto seq  = inputs.at(0);
    auto w    = inputs.at(1);
    auto r    = inputs.at(2);
    auto bias = inputs.at(3);
    auto ih   = inputs.at(4);

    instruction_ref hidden_states = prog.end(), last_output;
    long seq_len                  = static_cast<long>(seq->get_shape().lens()[0]);
    long hs                       = static_cast<long>(r->get_shape().lens()[2]);

    migraphx::shape s(seq->get_shape().type(),
                      {seq->get_shape().lens()[1], static_cast<std::size_t>(hs)});
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
    instruction_ref brcst_bz{};
    instruction_ref brcst_br{};
    instruction_ref brcst_wbh{};
    instruction_ref brcst_rbh{};
    instruction_ref brcst_bh{};
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
        auto xt = prog.insert_instruction(ins, op::slice{{0}, {seq_index}, {seq_index + 1}}, seq);
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

        if(i < seq_len - 1)
        {
            if(is_forward)
            {
                hidden_states =
                    (seq_index == 0)
                        ? last_output
                        : prog.insert_instruction(ins, op::concat{0}, hidden_states, last_output);
            }
            else
            {
                hidden_states =
                    (seq_index == seq_len - 1)
                        ? last_output
                        : prog.insert_instruction(ins, op::concat{0}, last_output, hidden_states);
            }
        }
    }

    return {hidden_states, last_output};
}

std::vector<operation> rewrite_rnn::gru_actv_funcs(instruction_ref ins) const
{
    auto gru_op = any_cast<op::gru>(ins->get_operator());
    // before rewrite the gru operator, need to ensure
    // we have 4 actv funcs, even though a user does not
    // specifiy any actv func. If less than 4, use the
    // algorithm in parse_gru to make 4 actv functions
    if(gru_op.direction == op::gru::bidirectional)
    {
        if(gru_op.actv_funcs.empty())
            return {op::sigmoid{}, op::tanh{}, op::sigmoid{}, op::tanh{}};
        else if(gru_op.actv_funcs.size() == 1)
            return {gru_op.actv_funcs.at(0),
                    gru_op.actv_funcs.at(0),
                    gru_op.actv_funcs.at(0),
                    gru_op.actv_funcs.at(0)};
        else if(gru_op.actv_funcs.size() == 2)
            return {gru_op.actv_funcs.at(0),
                    gru_op.actv_funcs.at(1),
                    gru_op.actv_funcs.at(0),
                    gru_op.actv_funcs.at(1)};
        else if(gru_op.actv_funcs.size() == 3)
            return {gru_op.actv_funcs.at(0),
                    gru_op.actv_funcs.at(1),
                    gru_op.actv_funcs.at(2),
                    gru_op.actv_funcs.at(0)};
        else
            return gru_op.actv_funcs;
    }
    else
    {
        if(gru_op.actv_funcs.empty())
            return {op::sigmoid{}, op::tanh{}};
        else if(gru_op.actv_funcs.size() == 1)
            return {gru_op.actv_funcs.at(0), gru_op.actv_funcs.at(0)};
        else
            return gru_op.actv_funcs;
    }
}

// for lstm operators
void rewrite_rnn::apply_lstm(program& prog, instruction_ref ins) const
{
    assert(ins->name() == "lstm");
    auto args = ins->inputs();

    shape seq_shape         = args[0]->get_shape();
    std::size_t hidden_size = args[1]->get_shape().lens()[2];
    std::size_t batch_size  = seq_shape.lens()[1];
    shape::type_t type      = seq_shape.type();
    migraphx::shape ihc_shape{type, {1, batch_size, hidden_size}};
    std::vector<float> ihc_data(ih_shape.elements(), 0.0);

    migraphx::shape pph_shape{type, {1, 3 * hidden_size}};
    std::vector<float> ppl_data(pph_shape.elements(), 0.0);

    auto& actv_funcs                 = lstm_actv_funcs(ins);
    auto lstm_op                     = any_cast<op::lstm>(ins->get_operator());
    op::lstm::lstm_direction_t dirct = lstm_op.direction;

    instruction_ref last_output{};
    instruction_ref last_cell_output{};
    if(dirct == op::lstm::bidirectional)
    {
        // input weight matrix
        // input weight matrix
        auto w_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[1]);
        auto w_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[1]);

        // hidden state weight matrix
        auto r_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[2]);
        auto r_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[2]);

        // process bias
        instruction_ref bias_forward = prog.end();
        instruction_ref bias_reverse = prog.end();
        if(args.size() >= 4 && args[3]->name() != "undefined")
        {
            bias_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[3]);
            bias_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[3]);
        }

        // process intial hidden state, it is the 6th argument
        instruction_ref ih_forward{};
        instruction_ref ih_reverse{};
        if(args.size() >= 6 && args[5]->name() != "undefined")
        {
            ih_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[5]);
            ih_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[5]);
        }
        else
        {
            ih_forward = prog.add_literal(migraphx::literal{ihc_shape, ihc_data});
            ih_reverse = prog.add_literal(migraphx::literal{ihc_shape, ihc_data});
        }

        // process initial cell value
        instruction_ref ic_forward{};
        instruction_ref ic_reverse{};
        if(args.size() >= 7 && args[6]->name() != "undefined")
        {
            ic_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[6]);
            ic_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[6]);
        }
        else
        {
            ic_forward = prog.add_literal(migraphx::literal{ihc_shape, ihc_data});
            ic_reverse = prog.add_literal(migraphx::literal{ihc_shape, ihc_data});
        }

        // process weight of the peephole
        instruction_ref pph_forward{};
        instruction_ref pph_reverse{};
        if(args.size() == 8 && args[7]->name() != "undefined")
        {
            pph_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[7]);
            pph_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[7]);
        }
        else
        {
            pph_forward = prog.add_literal(migraphx::literal{pph_shape, pph_data});
            pph_reverse = prog.add_literal(migraphx::literal{pph_shape, pph_data});
        }

        auto ret_forward = lstm_cell(
            true,
            prog,
            ins,
            {args[0], w_forward, r_forward, bias_forward, ih_forward, ic_forward, pph_forward},
            lstm_op.input_forget,
            actv_funcs.at(0),
            actv_funcs.at(1),
            actv_funcs.at(2));

        auto ret_reverse = lstm_cell(
            false,
            prog,
            ins,
            {args[0], w_reverse, r_reverse, bias_reverse, ih_reverse, ic_reverse, pph_reverse},
            lstm_op.input_forget,
            actv_funcs.at(3),
            actv_funcs.at(4),
            actv_funcs.at(5));

        auto concat_output =
            prog.insert_instruction(ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
        last_output = prog.insert_instruction(ins, op::squeeze{{0}}, concat_output);

        // last cell output
        auto concat_cell_output =
            prog.insert_instruction(ins, op::concat{1}, ret_forward[2], ret_reverse[2]);
        last_cell_output = prog.insert_instruction(ins, squeeze{{0}}, concat_cell_output);

        // the following logic is to ensure the last instruction is a concat
        if(ret_forward[0] == prog.end())
        {
            prog.replace_instruction(ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
        }
        else
        {
            ret_forward[0] =
                prog.insert_instruction(ins, op::concat{0}, ret_forward[0], ret_forward[1]);
            ret_reverse[0] =
                prog.insert_instruction(ins, op::concat{0}, ret_reverse[1], ret_reverse[0]);
            prog.replace_instruction(ins, op::concat{1}, {ret_forward[0], ret_reverse[0]});
        }
    }
    else
    {
    }
}

std::vector<instruction_ref> rewrite_rnn::lstm_cell(bool is_forward,
                                                    program& prog,
                                                    instruction_ref ins,
                                                    std::vector<instruction_ref> inputs,
                                                    int input_forget,
                                                    const operation& actv_func1,
                                                    const operation& actv_func2,
                                                    const operation& actv_func3) const
{
    return {};
}

std::vector<operation> rewrite_rnn::lstm_actv_funcs(instruction_ref ins) const
{
    auto lstm_op = any_cast<op::lstm>(ins->get_operator());
    // before rewrite the lstm operator, need to ensure
    // we have 6 actv funcs, even though a user does not
    // specifiy any actv func. If less than 46, use the
    // algorithm in parse_lstm to make 6 actv functions
    const auto& actv_funcs     = lstm_op.actv_funcs;
    std::size_t num_actv_funcs = actv_funcs.size();
    if(lstm_op.direction == op::lstm::bidirectional)
    {
        switch(num_actv_funcs)
        {
        case 0:
            return {op::sigmoid{}, op::tanh{}, op::tanh{}, op::sigmoid{}, op::tanh{}, op::tanh{}};

        case 1:
            return {actv_funcs.at(0),
                    actv_funcs.at(0),
                    actv_funcs.at(0),
                    actv_funcs.at(0),
                    actv_funcs.at(0),
                    actv_funcs.at(0)};

        case 2:
            return {actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(1),
                    actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(1)};

        case 3:
            return {actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(2),
                    actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(2)};

        case 4:
            return {actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(2),
                    actv_funcs.at(3),
                    actv_funcs.at(3),
                    actv_funcs.at(3)};

        case 5:
            return {actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(2),
                    actv_funcs.at(3),
                    actv_funcs.at(4),
                    actv_funcs.at(4)};

        default: return actv_funcs;
        }
    }
    else
    {
        switch(num_actv_funcs)
        {
        case 0: return {op::sigmoid{}, op::tanh{}, op::tanh{}};

        case 1: return {actv_funcs.at(0), actv_funcs.at(0), actv_funcs.at(0)};

        case 2: return {actv_funcs.at(0), actv_funcs.at(1), actv_funcs.at(1)};

        default: return actv_funcs;
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
