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
    std::unordered_map<instruction_ref, instruction_ref> map_last_output;
    for(auto ins : iterator_for(prog))
    {
        // rewrite rnn operator
        if(ins->name() == "rnn")
        {
            // could be 3 to 6 inputs, but the 5th input is undefined in
            // pytorch exported onnx, and it is ignored by protobuf. So
            // for input arguments 5 and 6, we need to check the shape,
            // then based on the shape to judge the specific input info
            auto args = ins->inputs();

            shape seq_shape         = args[0]->get_shape();
            std::size_t hidden_size = args[1]->get_shape().lens()[1];
            std::size_t batch_size  = seq_shape.lens()[1];
            shape::type_t type      = seq_shape.type();
            migraphx::shape ih_shape{type, {1, batch_size, hidden_size}};
            std::vector<float> data(ih_shape.elements(), 0);

            auto rnn_op                    = any_cast<op::rnn>(ins->get_operator());
            op::rnn::rnn_direction_t dicrt = rnn_op.direction;
            if(dicrt == op::rnn::rnn_direction_t::bidirectional)
            {
                // input weight matrix
                auto w_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[1]);
                auto w_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[1]);

                // hidden state weight matrix
                auto r_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[2]);
                auto r_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[2]);

                // process bias
                instruction_ref bias_forward, bias_reverse;
                bias_forward = bias_reverse = prog.end();
                if(args.size() >= 4)
                {
                    bias_forward = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, args[3]);
                    bias_reverse = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, args[3]);
                }

                // process intial hidden state, it could be the 6th argument
                // or the 5th one (if the sequence len argument is ignored)
                instruction_ref ih_forward, ih_reverse;
                if(args.size() == 6 ||
                   (args.size() == 5 && args[4]->get_shape().lens().size() == 3))
                {
                    auto arg_ih = (args.size() == 6) ? args[5] : args[4];
                    ih_forward  = prog.insert_instruction(ins, op::slice{{0}, {0}, {1}}, arg_ih);
                    ih_reverse  = prog.insert_instruction(ins, op::slice{{0}, {1}, {2}}, arg_ih);
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
                                            rnn_op.actv_funcs.at(0));
                auto ret_reverse = rnn_cell(false,
                                            prog,
                                            ins,
                                            args[0],
                                            w_reverse,
                                            r_reverse,
                                            bias_reverse,
                                            ih_reverse,
                                            rnn_op.actv_funcs.at(1));

                auto concat_output =
                    prog.insert_instruction(ins, op::concat{1}, ret_forward[1], ret_reverse[1]);
                auto last_output = prog.insert_instruction(ins, op::squeeze{{0}}, concat_output);

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
                map_last_output[hidden_output] = last_output;
            }
            else
            {
                bool is_forward = (dicrt == op::rnn::rnn_direction_t::forward);
                // input weight matrix
                auto w = args[1];

                // hidden state weight matrix
                auto r = args[2];

                // process bias and initial hidden state
                instruction_ref bias = prog.end();
                if(args.size() >= 4)
                {
                    bias = args[3];
                }

                // process intial hidden state
                instruction_ref ih;
                if(args.size() == 6 ||
                   (args.size() == 5 && args[4]->get_shape().lens().size() == 3))
                {
                    ih = (args.size() == 6) ? args[5] : args[4];
                }
                else
                {
                    ih = prog.add_literal(migraphx::literal{ih_shape, data});
                }

                auto ret = rnn_cell(
                    is_forward, prog, ins, args[0], w, r, bias, ih, rnn_op.actv_funcs.at(0));
                auto last_output = prog.insert_instruction(ins, op::squeeze{{0}}, ret[1]);

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
                map_last_output[hidden_output] = last_output;
            }
        }

        // rewrite the rnn_last_output operator that right after the rnn
        // operator. Intuitively, we can do a slice on its input to get
        // the last output, but it is already existed in the rnn operator,
        // so we can just use it as the output here
        if(ins->name() == "rnn_last_output")
        {
            auto inputs = ins->inputs();
            assert(inputs.size() == 1);
            auto arg = inputs[0];
            if(map_last_output.count(arg) == 0)
            {
                MIGRAPHX_THROW("RNN_LAST_OUTPUT: no related rnn operator as its input");
            }

            prog.replace_instruction(ins, map_last_output[arg]);
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

    instruction_ref hidden_out = prog.end(), last_out;
    last_out                   = prog.insert_instruction(ins, op::unsqueeze{{0, 1}}, sih);
    std::size_t seq_len        = input->get_shape().lens()[0];
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

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
