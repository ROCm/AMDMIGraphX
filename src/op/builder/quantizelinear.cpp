/* The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/quantize_dequantize_linear.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct quantizelinear : op_builder<quantizelinear>
{
    int axis       = 1;
    int block_size = 0;
    std::optional<migraphx::shape::type_t> output_type;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"),
                    f(self.block_size, "block_size"),
                    f(self.output_type, "output_type"));
    }

    std::vector<instruction_ref> handle_fp4x2(module& m,
                                              const std::vector<instruction_ref>& args) const
    {
        // Parsing in pack_fp4 and unpack_fp4 for the FP4 case
        auto q_ins = m.add_instruction(
            make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}), args);

        // packing axis set to fastest dimension
        auto quantized_shape   = q_ins->get_shape();
        const auto& qs_strides = quantized_shape.strides();
        if(qs_strides.empty())
        {
            MIGRAPHX_THROW("QuantizeLinear: MX type quantized_shape has no strides");
        }
        int fast_axis =
            std::min_element(qs_strides.cbegin(), qs_strides.cend()) - qs_strides.cbegin();
        bool odd_fast_axis = (quantized_shape.lens().at(fast_axis) % 2 == 1);
        if(odd_fast_axis)
        {
            // pad fastest dimension by 1 if it is odd
            std::vector<int64_t> padding(2 * quantized_shape.ndim(), 0);
            padding.at(fast_axis * 2 + 1) = 1;
            q_ins = m.add_instruction(make_op("pad", {{"pads", padding}}), q_ins);
        }
        // output is fp4x2_type
        auto pack_ins = m.add_instruction(make_op("pack_fp4"), q_ins);
        // output is fp8e4m3fn_type
        auto unpack_ins = m.add_instruction(make_op("unpack_fp4"), pack_ins);
        if(odd_fast_axis)
        {
            // slice off padded values
            unpack_ins =
                m.add_instruction(make_op("slice",
                                          {{"axes", {fast_axis}},
                                           {"starts", {0}},
                                           {"ends", {quantized_shape.lens().at(fast_axis)}}}),
                                  unpack_ins);
        }
        return {unpack_ins};
    }

    void convert_arg_to_common_type(module& m, std::vector<instruction_ref>& args) const
    {
        auto common_type = common_shape({args[0]->get_shape(), args[1]->get_shape()}).type();
        std::transform(args.begin(), args.begin() + 2, args.begin(), [&](auto ins) {
            if(ins->get_shape().type() != common_type)
                ins = m.add_instruction(make_op("convert", {{"target_type", common_type}}), ins);
            return ins;
        });
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref /*ins*/, const std::vector<instruction_ref>& args) const
    {
        auto args_new =
            transform_quantize_dequantize_linear_inputs(m, name(), block_size, axis, args);

        if(output_type == migraphx::shape::fp4x2_type)
        {
            return handle_fp4x2(m, args_new);
        }

        convert_arg_to_common_type(m, args_new);

        if(output_type.has_value())
            return {m.add_instruction(make_op("quantizelinear", {{"out_type", *output_type}}),
                                      args_new)};
        else
            return {m.add_instruction(make_op("quantizelinear"), args_new)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
