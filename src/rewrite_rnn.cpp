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
        // rewrite rnn operator
        if(ins->name() == "rnn")
        {
            // could be 3 to 6 inputs, but the parse_rnn function will
            // append undefined operators to make 6 arguments when parsing
            // an onnx file. Another case is user can have only 3 arguments
            // when writing their program.
            auto args = ins->inputs();

            shape seq_shape         = args[0]->get_shape();
            std::size_t hidden_size = args[1]->get_shape().lens()[1];
            std::size_t batch_size  = seq_shape.lens()[1];
            shape::type_t type      = seq_shape.type();
            migraphx::shape ih_shape{type, {1, batch_size, hidden_size}};
            std::vector<float> data(ih_shape.elements(), 0);

            auto actv_funcs                = compute_actv_funcs(ins);
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
                if(args.size() >= 4 && args[3]->get_operator().name() != "undefined")
                {
                    bias_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[3]);
                    bias_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[3]);
                }

                // process intial hidden state, it could be the 6th argument
                // or the 5th one (if the sequence len argument is ignored)
                instruction_ref ih_forward{};
                instruction_ref ih_reverse{};
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

                auto ret_forward = rnn_cell(true,
                                            prog,
                                            ins,
                                            args[0],
                                            w_forward,
                                            r_forward,
                                            bias_forward,
                                            ih_forward,
                                            actv_funcs.at(0));
                auto ret_reverse = rnn_cell(false,
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
                    hidden_output = prog.replace_instruction(
                        ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
                }
                else
                {
                    ret_forward[0] =
                        prog.insert_instruction(ins, op::concat{0}, ret_forward[0], ret_forward[1]);
                    ret_reverse[0] =
                        prog.insert_instruction(ins, op::concat{0}, ret_reverse[1], ret_reverse[0]);
                    hidden_output = prog.replace_instruction(
                        ins, op::concat{1}, {ret_forward[0], ret_reverse[0]});
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
                if(args.size() >= 4 && args[3]->get_operator().name() != "undefined")
                {
                    bias = args[3];
                }

                // process intial hidden state
                instruction_ref ih;
                if(args.size() == 6 && args[5]->get_operator().name() != "undefined")
                {
                    ih = args[5];
                }
                else
                {
                    ih = prog.add_literal(migraphx::literal{ih_shape, data});
                }

                auto ret =
                    rnn_cell(is_forward, prog, ins, args[0], w, r, bias, ih, actv_funcs.at(0));
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
                    hidden_output =
                        prog.replace_instruction(ins, op::concat{0}, concat_arg0, concat_arg1);
                }
            }

            // search its output to find if there are rnn_last_output operator
            // while loop to handle case of multiple rnn_last_output operators
            auto last_output_it = ins->outputs().begin();
            while (last_output_it != ins->outputs().end())
            {
                last_output_it = std::find_if(last_output_it, ins->outputs().end(), [] (auto i) {
                    return i->name() == "rnn_last_output";
                });

                if(last_output_it != ins->outputs().end())
                {
                    prog.replace_instruction(*last_output_it, last_output);
                    last_output_it++;
                }
            }
        }
    }
}

std::vector<instruction_ref> rewrite_rnn::rnn_cell(bool is_forward,
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

std::vector<operation> rewrite_rnn::compute_actv_funcs(instruction_ref ins) const
{
    auto rnn_op = any_cast<op::rnn>(ins->get_operator());
    // before rewrite the rnn operator, need to ensure
    // we have 2 actv funcs. If less than 2, use the
    // algorithm in parse_rnn to make 2 actv functions
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

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
