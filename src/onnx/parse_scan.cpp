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
#include "migraphx/errors.hpp"
#include "migraphx/instruction_ref.hpp"
#include <algorithm>
#include <cstdint>
#include <iterator>
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
        // NOTE Version 8 of the operator differs to all the later versions
        if(not contains(info.attributes, "body"))
            MIGRAPHX_THROW("Scan: body attribute required");

        if(not contains(info.attributes, "num_scan_inputs"))
            MIGRAPHX_THROW("Scan: num_scan_inputs attribute required");

        const auto& body = info.attributes["body"].g();
        auto sub_mod     = parser.prog.create_module(info.name + "_scan");
        (void)parser.parse_graph(sub_mod, body);

        const auto num_scan_inputs = info.attributes["num_scan_inputs"].i();
        auto N                     = args.size() - num_scan_inputs;
        auto sub_mod_output_shapes = sub_mod->get_output_shapes();
        auto K                     = sub_mod_output_shapes.size() - N;

        std::vector<int64_t> scan_input_axes(num_scan_inputs, 0);
        if(contains(info.attributes, "scan_input_axes"))
        {
            auto&& axes = info.attributes["scan_input_axes"].ints();
            scan_input_axes.assign(axes.begin(), axes.end());
            // Validate: Size of scan_input_axes must be equal to num_scan_inputs
            // Perform: Normalize the axes
        }
        // Validate: The scan axis len across each scan_in must be equal

        // TODO
        // Parse scan_input_directions
        // Validate: Size of scan_input_directions must be equal to num_scan_inputs
        // Validate: 0 and 1 are only allowed values

        // SCAN OUTPUT AXES
        std::vector<int64_t> scan_output_axes(K, 0);
        if(contains(info.attributes, "scan_output_axes"))
        {
            auto&& axes = info.attributes["scan_output_axes"].ints();
            scan_output_axes.assign(axes.begin(), axes.end());

            if(scan_output_axes.size() != K)
                MIGRAPHX_THROW("Number of scan output axes (" + to_string(scan_output_axes.size()) +
                               ") does not match number of body scan outputs(" + to_string(K) +
                               ")");

            std::vector<int64_t> ndims;
            ndims.reserve(K);
            std::transform(sub_mod_output_shapes.begin() + N,
                           sub_mod_output_shapes.end(),
                           std::back_inserter(ndims),
                           [](const shape& sh) { return sh.ndim() + 1; });
            normalize_axes(scan_output_axes, ndims);
        }
        std::cout << to_string_range(scan_output_axes) << std::endl;
        // SCAN OUTPUT AXES

        // SCAN OUTPUT DIRECTIONS
        std::vector<int64_t> scan_output_directions(K, 0);
        if(contains(info.attributes, "scan_output_directions"))
        {
            auto&& dirs = info.attributes["scan_output_directions"].ints();
            scan_output_directions.assign(dirs.begin(), dirs.end());

            if(scan_output_directions.size() != K)
                MIGRAPHX_THROW("Number of scan output directions (" +
                               to_string(scan_output_directions.size()) +
                               ") does not match number of body scan outputs(" + to_string(K) +
                               ")");

            if(any_of(scan_output_directions, [](auto i) { return i != 0 and i != 1; }))
                MIGRAPHX_THROW(
                    "Scan output directions may contain only 1s and 0s, actual values: " +
                    to_string_range(scan_output_directions));
        }
        // SCAN OUTPUT DIRECTIONS

        size_t num_iters = args[N]->get_shape().lens()[scan_input_axes[0]];
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

        // Inputs: init_states, array of pre-sliced scan_inputs
        // N + M * num_iters number of inputs
        auto scan = info.add_instruction(make_op("scan",
                                                 {{"iterations", num_iters},
                                                  {"num_scan_inputs", num_scan_inputs},
                                                  {"num_state_vars", N}}),
                                         alt_args,
                                         {sub_mod});
        // Outputs: final_states, array of scan_output_elements
        // N + K * num_iters number of outputs

        std::vector<instruction_ref> ret;
        for(auto i = 0; i < N; ++i)
        {
            auto get = info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), scan);
            ret.push_back(get);
        }

        for(auto i = N; i < N + K; ++i)
        {
            auto get = info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), scan);
            auto scan_axis = scan_output_axes[i - N];
            auto usq = info.add_instruction(make_op("unsqueeze", {{"axes", {scan_axis}}}), get);
            ret.push_back(usq);
        }

        for(auto i = 1; i < num_iters; ++i)
        {
            for(auto j = 0; j < K; ++j)
            {
                auto tuple_idx = N + i * K + j;
                auto get =
                    info.add_instruction(make_op("get_tuple_elem", {{"index", tuple_idx}}), scan);
                auto scan_axis = scan_output_axes[j];
                auto usq = info.add_instruction(make_op("unsqueeze", {{"axes", {scan_axis}}}), get);
                auto dir = scan_output_directions[j];
                std::vector<instruction_ref> concat_args(2, usq);
                concat_args[dir] = ret[N + j];
                auto concat =
                    info.add_instruction(make_op("concat", {{"axis", scan_axis}}), concat_args);
                ret[N + j] = concat;
            }
        }

        return ret;
    }

    void normalize_axes(std::vector<int64_t>& axes, const std::vector<int64_t>& ndims) const
    {
        auto normalize_axis = [=](int64_t axis, int64_t ndim) {
            if(axis < -ndim or axis >= ndim)
                MIGRAPHX_THROW("Axis value {" + to_string(axis) + "} out of range [" +
                               to_string(-ndim) + ", " + to_string(ndim) + ")");

            return axis < 0 ? ndim + axis : axis;
        };

        std::transform(axes.begin(), axes.end(), ndims.begin(), axes.begin(), normalize_axis);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
