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
        if(ins->name() != "rnn")
        {
            continue;
        }

        // could be 3 to 5 inputs (though onnx::rnn has 6 inputs,
        // the 5th one is undefined and ignored by protobuf. so
        // we need to process up to 5 inputs
        auto args = ins->inputs();

        shape seq_shape         = args[0]->get_shape();
        shape wgt_shape         = args[1]->get_shape();
        std::size_t hidden_size = wgt_shape.lens()[1];
        std::size_t batch_size  = seq_shape.lens()[1];
        shape::type_t type      = seq_shape.type();
        migraphx::shape s{type, {batch_size, hidden_size}};
        std::vector<char> data(s.bytes(), 0);

        auto rnn_op                    = any_cast<op::rnn>(ins->get_operator());
        op::rnn::rnn_direction_t dicrt = rnn_op.direction;
        if(dicrt == op::rnn::rnn_direction_t::bidirectional)
        {
            std::vector<int64_t> perm{1, 0};
            // process input weight matrix
            // forward
            auto xw_forward       = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[1]);
            auto sxw_forward      = prog.insert_instruction(ins, op::squeeze{{0}}, xw_forward);
            auto trans_xw_forward = prog.insert_instruction(ins, op::transpose{perm}, sxw_forward);

            // reverse
            auto xw_reverse       = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[1]);
            auto sxw_reverse      = prog.insert_instruction(ins, op::squeeze{{0}}, xw_reverse);
            auto trans_xw_reverse = prog.insert_instruction(ins, op::transpose{perm}, sxw_reverse);

            // process hidden state weight matrix
            auto hw_forward       = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[2]);
            auto shw_forward      = prog.insert_instruction(ins, op::squeeze{{0}}, hw_forward);
            auto trans_hw_forward = prog.insert_instruction(ins, op::transpose{perm}, shw_forward);

            auto hw_reverse       = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[2]);
            auto shw_reverse      = prog.insert_instruction(ins, op::squeeze{{0}}, hw_reverse);
            auto trans_hw_reverse = prog.insert_instruction(ins, op::transpose{perm}, shw_reverse);

            // process bias
            instruction_ref bias_forward, bias_reverse;
            bias_forward = bias_reverse = prog.end();
            if(args.size() >= 4)
            {
                // forward
                long h_size    = static_cast<long>(hidden_size);
                auto b_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[3]);
                b_forward      = prog.insert_instruction(ins, op::squeeze{{0}}, b_forward);
                auto wbf = prog.insert_instruction(ins, op::slice{{0}, {0}, {h_size}}, b_forward);
                auto rbf =
                    prog.insert_instruction(ins, op::slice{{0}, {h_size}, {2 * h_size}}, b_forward);
                auto bf      = prog.insert_instruction(ins, op::add{}, wbf, rbf);
                bias_forward = prog.insert_instruction(ins, op::broadcast{1, s}, bf);

                // backward
                auto b_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[3]);
                b_reverse      = prog.insert_instruction(ins, op::squeeze{{0}}, b_reverse);
                auto wbr = prog.insert_instruction(ins, op::slice{{0}, {0}, {h_size}}, b_reverse);
                auto rbr =
                    prog.insert_instruction(ins, op::slice{{0}, {h_size}, {2 * h_size}}, b_reverse);
                auto br      = prog.insert_instruction(ins, op::add{}, wbr, rbr);
                bias_reverse = prog.insert_instruction(ins, op::broadcast{1, s}, br);
            }

            // process intial hidden state
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
                ih_forward = prog.add_literal(migraphx::literal{s, data});
                ih_reverse = prog.add_literal(migraphx::literal{s, data});
            }

            auto ret_forward = rnn_oper(true,
                                        prog,
                                        ins,
                                        args[0],
                                        trans_xw_forward,
                                        trans_hw_forward,
                                        ih_forward,
                                        bias_forward,
                                        rnn_op.actv_funcs.at(0));
            auto ret_reverse = rnn_oper(false,
                                        prog,
                                        ins,
                                        args[0],
                                        trans_xw_reverse,
                                        trans_hw_reverse,
                                        ih_reverse,
                                        bias_reverse,
                                        rnn_op.actv_funcs.at(1));

            // auto final_output = prog.insert_instruction(ins, op::concat{0}, ret_forward[1],

            // add the dimension of num_direction
            ret_forward[0] = prog.insert_instruction(ins, op::unsqueeze{{1}}, ret_forward[0]);
            ret_reverse[0] = prog.insert_instruction(ins, op::unsqueeze{{1}}, ret_reverse[0]);

            // concat the forward and reverse output
            prog.replace_instruction(ins, op::concat{1}, {ret_forward[0], ret_reverse[0]});
        }
        else
        {
            bool is_forward = (dicrt == op::rnn::rnn_direction_t::forward) ? true : false;
            std::vector<int64_t> perm{1, 0};
            // process input weight matrix
            auto sxw      = prog.insert_instruction(ins, op::squeeze{{0}}, args[1]);
            auto trans_xw = prog.insert_instruction(ins, op::transpose{perm}, sxw);

            // process hidden state weight matrix
            auto shw      = prog.insert_instruction(ins, op::squeeze{{0}}, args[2]);
            auto trans_hw = prog.insert_instruction(ins, op::transpose{perm}, shw);

            // process bias and initial hidden state
            instruction_ref bias = prog.end();
            if(args.size() >= 4)
            {
                long h_size = static_cast<long>(hidden_size);
                auto bwr    = prog.insert_instruction(ins, op::squeeze{{0}}, args[3]);
                auto wb     = prog.insert_instruction(ins, op::slice{{0}, {0}, {h_size}}, bwr);
                auto rb = prog.insert_instruction(ins, op::slice{{0}, {h_size}, {2 * h_size}}, bwr);
                auto b  = prog.insert_instruction(ins, op::add{}, wb, rb);
                bias    = prog.insert_instruction(ins, op::broadcast{1, s}, b);
            }

            // process intial hidden state
            instruction_ref ih;
            if(args.size() >= 5)
            {
                ih = prog.insert_instruction(ins, op::squeeze{{0}}, args[4]);
            }
            else
            {
                ih = prog.add_literal(migraphx::literal{s, data});
            }
            auto ret = rnn_oper(
                is_forward, prog, ins, args[0], trans_xw, trans_hw, ih, bias, rnn_op.actv_funcs.at(0));

            // add the dimension of num_direction
            prog.replace_instruction(ins, op::unsqueeze{{1}}, ret[0]);
        }
    }
}

std::vector<instruction_ref> rewrite_rnn::rnn_oper(bool is_forward,
                                                   program& prog,
                                                   instruction_ref ins,
                                                   instruction_ref input,
                                                   instruction_ref wx,
                                                   instruction_ref wh,
                                                   instruction_ref ih,
                                                   instruction_ref bias,
                                                   operation& actv_func) const
{
    instruction_ref hidden_out, final_out;
    migraphx::shape input_shape = input->get_shape();
    std::size_t seq_len         = input_shape.lens()[0];
    long seq_index              = is_forward ? 0 : seq_len - 1;
    for(std::size_t i = 0; i < seq_len; i++)
    {
        auto xt = prog.insert_instruction(ins, op::slice{{0}, {seq_index}, {seq_index + 1}}, input);
        xt      = prog.insert_instruction(ins, op::squeeze{{0}}, xt);
        auto x_w = prog.insert_instruction(ins, op::dot{}, xt, wx);
        auto h_r = prog.insert_instruction(ins, op::dot{}, ih, wh);
        auto x_h = prog.insert_instruction(ins, op::add{}, x_w, h_r);
        instruction_ref before_actv;
        if(bias != prog.end())
        {
            before_actv = prog.insert_instruction(ins, op::add{}, x_h, bias);
        }
        else
        {
            before_actv = x_h;
        }

        // apply activation function
        ih = prog.insert_instruction(ins, actv_func, before_actv);

        // add the dimension of sequence length
        auto output = prog.insert_instruction(ins, op::unsqueeze{{0}}, ih);
        final_out   = output;

        if(is_forward)
        {
            hidden_out = (seq_index == 0)
                             ? output
                             : prog.insert_instruction(ins, op::concat{0}, hidden_out, output);
        }
        else
        {
            hidden_out = (seq_index == seq_len - 1)
                             ? output
                             : prog.insert_instruction(ins, op::concat{0}, output, hidden_out);
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
