/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <random>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_multinomial : op_parser<parse_multinomial>
{
    std::vector<op_desc> operators() const { return {{"Multinomial"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        if(args.empty())
            MIGRAPHX_THROW("PARSE_MULTINOMIAL: no arguments given");

        int dtype = 6;
        if(contains(info.attributes, "dtype"))
            dtype = info.attributes.at("dtype").i();
        shape::type_t output_type = get_type(dtype);

        size_t sample_size = 1;
        if(contains(info.attributes, "sample_size"))
            sample_size = info.attributes.at("sample_size").i();
        else
            MIGRAPHX_THROW("PARSE_MULTINOMIAL: sample_size not given");

        // Subtract the per-batch maximum log-probability, making the per-batch max 0
        auto maxes =
            info.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), args[0]);
        auto cdf = info.add_common_op("sub", args[0], maxes);
        // Take the element-wise exponent to get probabilities in the range (0, 1]
        // cdf = info.add_instruction(migraphx::make_op("exp"), cdf);
        // Compute the cumulative distribution function
        cdf = info.add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

        instruction_ref seed_input;
        if(contains(info.attributes, "seed"))
        {
            uint32_t seed = info.attributes.at("seed").i();
            migraphx::shape s{migraphx::shape::uint32_type, {1}};
            std::vector<uint32_t> data = {seed};
            seed_input                 = info.add_literal(migraphx::literal(s, data));
        }
        else
        {
            seed_input = info.add_instruction(migraphx::make_op("random_seed"));
        }
        instruction_ref randoms;

        shape s0 = args[0]->get_shape();

        if(s0.dynamic())
        {
            //  Dynamic batch_size will be taken from args[0].  The input argument to this should
            // have a second dimension of sample_size.
            std::vector<shape::dynamic_dimension> dyn_dim_set;
            dyn_dim_set.emplace_back(s0.dyn_dims().front());
            dyn_dim_set.emplace_back(shape::dynamic_dimension{sample_size, sample_size});
            auto alloc = info.add_instruction(migraphx::make_op("allocate", {{"shape", to_value(shape{s0.type(), dyn_dim_set})}}));
            randoms =
                info.add_instruction(migraphx::make_op("random_uniform"), seed_input, alloc);
        }
        else
        {
            // use literal.  It may be quite large.
            size_t batch_size = s0.lens().front();
            auto rand_dummy   = info.add_literal(
                migraphx::literal{migraphx::shape::float_type, {batch_size * sample_size}});

            randoms =
                info.add_instruction(migraphx::make_op("random_uniform"), seed_input, rand_dummy);
        }

        migraphx::shape asdf = randoms->get_shape();

        return info.add_instruction(
            migraphx::make_op("multinomial", {{"dtype", output_type}}), cdf, randoms);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
