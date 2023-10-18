/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/map_activation_functions.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

void lstm_actv_functions(op::rnn_direction dirct, std::vector<std::string>& actv_func_names)
{
    // need 6 activation functions for bidirectional directions
    if(dirct == op::rnn_direction::bidirectional)
    {
        // 6 activation functions are used in the bidirectional
        // scenario. No spec is provided in onnx::operator. we
        // use the algorithm that: if 1 actv function is provided,
        // repeat 1st six times. If 2 actv functins are provided,
        // repeat 2nd once, then repeat all three once
        // if 3 actv funcs are provide, repeat all three once.
        // the same algorithm is used for 4, 5, and 6 actv funcions
        // provided. This may need change later
        switch(actv_func_names.size())
        {
        case 1:
            actv_func_names = {actv_func_names.at(0),
                               actv_func_names.at(0),
                               actv_func_names.at(0),
                               actv_func_names.at(0),
                               actv_func_names.at(0),
                               actv_func_names.at(0)};
            break;

        case 2:
            // repeat the 2nd actv func once, then repeat all three another time
            actv_func_names = {actv_func_names.at(0),
                               actv_func_names.at(1),
                               actv_func_names.at(1),
                               actv_func_names.at(0),
                               actv_func_names.at(1),
                               actv_func_names.at(1)};
            break;

        case 3:
            // repeat all three actv funcs once
            actv_func_names = {actv_func_names.at(0),
                               actv_func_names.at(1),
                               actv_func_names.at(2),
                               actv_func_names.at(0),
                               actv_func_names.at(1),
                               actv_func_names.at(2)};
            break;

        case 4:
            actv_func_names = {actv_func_names.at(0),
                               actv_func_names.at(1),
                               actv_func_names.at(2),
                               actv_func_names.at(3),
                               actv_func_names.at(3),
                               actv_func_names.at(3)};
            break;

        case 5:
            actv_func_names = {actv_func_names.at(0),
                               actv_func_names.at(1),
                               actv_func_names.at(2),
                               actv_func_names.at(3),
                               actv_func_names.at(4),
                               actv_func_names.at(4)};
            break;

        default: break;
        }
    }
    else
    {
        switch(actv_func_names.size())
        {
        case 1:
            actv_func_names = {actv_func_names.at(0), actv_func_names.at(0), actv_func_names.at(0)};
            break;

        case 2:
            // repeat the 2nd actv func once, so we have 3 actv funcs
            actv_func_names = {actv_func_names.at(0), actv_func_names.at(1), actv_func_names.at(1)};
            break;

        default: break;
        }
    }
}

struct parse_lstm : op_parser<parse_lstm>
{
    std::vector<op_desc> operators() const { return {{"LSTM"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       std::vector<instruction_ref> args) const
    {
        migraphx::shape input_shape = args[0]->get_shape();
        std::size_t hidden_size     = args[2]->get_shape().lens()[2];

        if(contains(info.attributes, "hidden_size"))
        {
            std::size_t hidden_size_att =
                parser.parse_value(info.attributes.at("hidden_size")).at<int>();
            if(hidden_size != hidden_size_att)
            {
                MIGRAPHX_THROW("LSTM: hidden size mismatch in input and attribute");
            }
        }

        // Handling of direction to be added later
        std::string direction{"forward"};
        if(contains(info.attributes, "direction"))
        {
            direction = info.attributes.at("direction").s();
        }

        op::rnn_direction dirct = op::rnn_direction::forward;
        if(direction == "bidirectional")
        {
            dirct = op::rnn_direction::bidirectional;
        }
        else if(direction == "reverse")
        {
            dirct = op::rnn_direction::reverse;
        }
        else if(direction == "forward")
        {
            dirct = op::rnn_direction::forward;
        }
        else
        {
            MIGRAPHX_THROW("LSTM: incorrect direction attribute");
        }

        std::vector<std::string> vec_names = {"sigmoid", "tanh", "tanh"};
        if(contains(info.attributes, "activations"))
        {
            auto names = info.attributes.at("activations").strings();
            vec_names.clear();
            vec_names.resize(names.size());
            std::transform(names.begin(), names.end(), vec_names.begin(), [](auto name) {
                return to_lower(name);
            });
        }

        lstm_actv_functions(dirct, vec_names);

        auto name_it = std::find_if(vec_names.begin(), vec_names.end(), [&](auto& name) {
            return (map_activation_functions().count(name) == 0);
        });
        if(name_it != vec_names.end())
        {
            MIGRAPHX_THROW("LSTM: activation function " + std::string(*name_it) + " not supported");
        }

        std::vector<operation> vec_actv_funcs(vec_names.size());
        std::transform(vec_names.begin(),
                       vec_names.end(),
                       vec_actv_funcs.begin(),
                       [&](const auto& name) { return map_activation_functions().at(name); });

        float clip = 0.0;
        if(contains(info.attributes, "clip"))
        {
            clip = parser.parse_value(info.attributes.at("clip")).at<float>();
        }

        int input_forget = 0;
        if(contains(info.attributes, "input_forget"))
        {
            input_forget = parser.parse_value(info.attributes.at("input_forget")).at<int>();
        }

        int layout = 0;
        if(contains(info.attributes, "layout"))
        {
            layout = parser.parse_value(info.attributes.at("layout")).at<int>();
        }

        // append undefined opeator to make 6 arguments
        if(args.size() < 8)
        {
            auto ins = info.add_instruction(make_op("undefined"));
            args.insert(args.end(), 8 - args.size(), ins);
        }

        // first output for concatenation of hidden states
        auto hidden_states = info.add_instruction(make_op("lstm",
                                                          {{"hidden_size", hidden_size},
                                                           {"actv_func", to_value(vec_actv_funcs)},
                                                           {"direction", dirct},
                                                           {"clip", clip},
                                                           {"input_forget", input_forget},
                                                           {"layout", layout}}),
                                                  args);

        auto last_output = info.add_instruction(make_op("rnn_last_hs_output", {{"layout", layout}}),
                                                hidden_states);

        // third output for last cell output
        auto last_cell_output = info.add_instruction(
            make_op("rnn_last_cell_output", {{"layout", layout}}), hidden_states);

        return {hidden_states, last_output, last_cell_output};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
