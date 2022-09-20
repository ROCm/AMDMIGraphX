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

struct parse_rnn : op_parser<parse_rnn>
{
    std::vector<op_desc> operators() const { return {{"RNN"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       std::vector<instruction_ref> args) const
    {
        migraphx::shape input_shape = args[0]->get_shape();
        std::size_t hidden_size     = args[1]->get_shape().lens()[1];

        if(contains(info.attributes, "hidden_size"))
        {
            std::size_t hidden_size_att =
                parser.parse_value(info.attributes.at("hidden_size")).at<int>();
            if(hidden_size != hidden_size_att)
            {
                MIGRAPHX_THROW("RNN: hidden size mismatch in input and attribute");
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

        std::vector<std::string> vec_names{"tanh"};
        if(contains(info.attributes, "activations"))
        {
            auto names = info.attributes.at("activations").strings();
            vec_names.clear();
            vec_names.resize(names.size());
            std::transform(names.begin(), names.end(), vec_names.begin(), [](auto name) {
                return to_lower(name);
            });
        }

        auto name_it = std::find_if(vec_names.begin(), vec_names.end(), [&](auto& name) {
            return (map_activation_functions().count(name) == 0);
        });
        if(name_it != vec_names.end())
        {
            MIGRAPHX_THROW("RNN: activation function " + std::string(*name_it) + " not supported");
        }

        // bidirectional case should have two activation functions.
        // one is for forward, and the other is for reverse.
        // if only one actv function is provided, we use it in both
        // forward and reverse direction
        if(dirct == op::rnn_direction::bidirectional)
        {
            if(vec_names.size() == 1)
            {
                vec_names.push_back(vec_names.at(0));
            }
        }

        std::vector<operation> vec_actv_funcs(vec_names.size());
        std::transform(vec_names.begin(),
                       vec_names.end(),
                       vec_actv_funcs.begin(),
                       [&](const auto& fn) { return map_activation_functions().at(fn); });

        // To be added later
        float clip = 0.0;
        if(contains(info.attributes, "clip"))
        {
            clip = parser.parse_value(info.attributes.at("clip")).at<float>();
        }

        // if the number of arguments is less than 6, append
        // undefined operator to have 6 arguments
        if(args.size() < 6)
        {
            auto ins = info.add_instruction(make_op("undefined"));
            args.insert(args.end(), (6 - args.size()), ins);
        }

        // first output for the concatenation of hidden states
        auto hidden_states = info.add_instruction(make_op("rnn",
                                                          {{"hidden_size", hidden_size},
                                                           {"actv_func", to_value(vec_actv_funcs)},
                                                           {"direction", dirct},
                                                           {"clip", clip}}),
                                                  args);

        // second output for the last hidden state
        auto last_output = info.add_instruction(make_op("rnn_last_hs_output"), hidden_states);

        return {hidden_states, last_output};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
