/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_scan : op_parser<parse_scan>
{
    std::vector<op_desc> operators() const { return {{"Scan"}}; }

    std::vector<instruction_ref> parse(const op_desc& opd,
                                       onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       std::vector<instruction_ref> args) const
    {
        std::cout << "Scan parse" << std::endl;
        if(not contains(info.attributes, "body"))
            MIGRAPHX_THROW("Scan: body attribute required");

        if(not contains(info.attributes, "num_scan_inputs"))
            MIGRAPHX_THROW("Scan: num_scan_inputs attribute required");

        const auto& body           = info.attributes["body"].g();
        const auto num_scan_inputs = info.attributes["num_scan_inputs"].i();

        auto sub_mod = parser.prog.create_module(info.name + "_scan");
        (void)parser.parse_graph(sub_mod, body);

        auto param_names = sub_mod->get_parameter_names();
        std::cout << param_names.size() << std::endl;
        std::cout << args.size() << std::endl;

        // This does not hold for the opset 8 version of Scan, which has an optional first input
        // that the other versions do not have
        // if(param_names.size() != args.size())
        //     MIGRAPHX_THROW("Scan: Number of inputs to Scan does not match the number of inputs to
        //     "
        //                    "its subgraph");

        std::vector<int64_t> scan_input_axes(num_scan_inputs, 0);
        if(contains(info.attributes, "scan_input_axes"))
        {
            auto&& axes = info.attributes["scan_input_axes"].ints();
            // if(axes.size() != num_scan_inputs)
            // {
            //     MIGRAPHX_THROW("Scan: Size of scan_input_axes needs to match num_scan_inputs");
            // }
            scan_input_axes.assign(axes.begin(), axes.end());
            // TODO add axes normalization
        }
        std::cout << "scan_input_axes: " << to_string_range(scan_input_axes) << std::endl;

        // TODO check that scan_input_axes lens match for every arg
        auto num_inits   = args.size() - num_scan_inputs;
        size_t num_iters = args[num_inits]->get_shape().lens()[scan_input_axes[0]];

        std::cout << "num_scan_inputs: " << num_scan_inputs << std::endl;
        std::cout << "args shapes: " << to_string_range(to_shapes(args)) << std::endl;
        sub_mod->debug_print();
        auto param_shapes = sub_mod->get_parameter_shapes();
        std::cout << to_string_range(param_names) << std::endl;
        for(const auto& s : param_shapes)
            std::cout << s.first << " " << s.second << std::endl;
        std::cout << "num_iters: " << num_iters << std::endl;

        auto N = args.size() - num_scan_inputs;
        std::vector<instruction_ref> alt_args(args.begin(), args.begin() + N);
        for(int64_t i = 0; i < num_iters; ++i)
        {
            std::transform(
                args.begin() + N, args.end(), std::back_inserter(alt_args), [&](const auto& arg) {
                    auto slice = info.add_instruction(
                        make_op("slice", {{"axes", {0}}, {"starts", {i}}, {"ends", {i + 1}}}), arg);
                    return info.add_instruction(make_op("squeeze", {{"axes", {0}}}), slice);
                });
        }

        std::cout << "Whole args" << std::endl;
        for(const auto& ins : args)
        {
            ins->debug_print();
        }
        std::cout << "Sliced args" << std::endl;
        for(const auto& ins : alt_args)
        {
            ins->debug_print();
        }

        auto ret = info.add_instruction(make_op("scan",
                                                {{"iterations", num_iters},
                                                 {"num_scan_inputs", num_scan_inputs},
                                                 {"num_state_vars", N}}),
                                        alt_args,
                                        {sub_mod});

        auto out_s = ret->get_shape();
        assert(out_s.type() == shape::tuple_type);

        const auto& vec_shapes = out_s.sub_shapes();
        std::vector<instruction_ref> out_inss;
        for(std::size_t i = 0; i < vec_shapes.size(); ++i)
        {
            auto r = info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), ret);
            out_inss.push_back(r);
        }

        return out_inss;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
