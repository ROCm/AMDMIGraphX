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
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Convert 2 input static shape broadcast/multibroadcast into 1 input version.
 * Some compiler passes (ex. simplify_algebra) only support the 1 input versions
 * of the broadcasting operators.
 * From:
 * broadcast_op(argument_with_static_shape, argument_with_static_shape)
 * To:
 * broadcast_op(argument_with_static_shape); broadcast_op.out_lens = constant_output_dims
 */
struct find_static_2in_broadcasts
{
    auto matcher() const
    {
        return match::broadcast(match::nargs(2),
                                match::arg(0)(match::static_shape()),
                                match::arg(1)(match::static_shape()));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins          = mr.result;
        auto out_lens     = ins->get_shape().lens();
        auto broadcast_op = ins->get_operator();
        if(broadcast_op.name() == "broadcast")
        {
            broadcast_op.from_value({{"out_lens", out_lens}});
        }
        else
        {
            broadcast_op.from_value({{"out_lens", out_lens}, {"out_dyn_dims", {}}});
        }
        m.replace_instruction(ins, broadcast_op, ins->inputs().at(0));
    }
};

/**
 * Simplify slice with 2 inputs to the 1 input version if inputs[1] is constant.
 * From:
 * slice(data, constant_input); two attributes set
 * To:
 * slice(data); slice.starts, slice.ends. slice.axes set
 */
struct find_const_2in_slice
{
};

/**
 * Simplify slice with 3 inputs to the 1 input version if inputs[1:2] are constant.
 * From:
 * slice(data, constant_input1, constant_input2); one attribute set
 * To:
 * slice(data); slice.starts, slice.ends. slice.axes set
 */
struct find_const_3in_slice
{
    auto matcher() const
    {
        return match::name("slice")(match::nargs(3),
                                    match::arg(1)(match::is_constant()),
                                    match::arg(2)(match::is_constant()));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins            = mr.result;
        auto inputs         = ins->inputs();
        auto slice_val      = ins->get_operator().to_value();
        auto slice_op       = any_cast<op::slice>(ins->get_operator());
        auto set_attrs      = slice_op.get_set_attributes();
        std::vector<int64_t> starts_vec;
        std::vector<int64_t> ends_vec;
        std::vector<int64_t> axes_vec;
        if(set_attrs == slice_op.axes_only)
        {
            // slice(data, starts, ends)
            inputs.at(1)->eval().visit(
                [&](auto output) { starts_vec.assign(output.begin(), output.end()); });
            inputs.at(2)->eval().visit(
                [&](auto output) { ends_vec.assign(output.begin(), output.end()); });
            axes_vec = slice_val.at("axes").to_vector<int64_t>();
        }
        else if(set_attrs == slice_op.ends_only)
        {
            // slice(data, starts, axes)
            inputs.at(1)->eval().visit(
                [&](auto output) { starts_vec.assign(output.begin(), output.end()); });
            inputs.at(2)->eval().visit(
                [&](auto output) { axes_vec.assign(output.begin(), output.end()); });
            ends_vec = slice_val.at("ends").to_vector<int64_t>();
        }
        else
        {
            // slice(data, ends, axes)
            inputs.at(1)->eval().visit(
                [&](auto output) { ends_vec.assign(output.begin(), output.end()); });
            inputs.at(2)->eval().visit(
                [&](auto output) { axes_vec.assign(output.begin(), output.end()); });
            starts_vec = slice_val.at("starts").to_vector<int64_t>();
        }
        m.replace_instruction(
            ins,
            make_op("slice", {{"starts", starts_vec}, {"ends", ends_vec}, {"axes", axes_vec}}),
            inputs.at(0));
    }
};

/**
 * Simplify slice with 4 inputs to the 1 input version if inputs[1:3] are constant.
 * From:
 * slice(data, constant_starts, constant_ends, constant_axes)
 * To:
 * slice(data); slice.starts, slice.ends. slice.axes set
 */
struct find_const_4in_slice
{
    auto matcher() const
    {
        return match::name("slice")(match::nargs(4),
                                    match::arg(1)(match::is_constant()),
                                    match::arg(2)(match::is_constant()),
                                    match::arg(3)(match::is_constant()));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins            = mr.result;
        auto inputs         = ins->inputs();
        argument starts_arg = inputs.at(1)->eval(false);
        argument ends_arg   = inputs.at(2)->eval(false);
        argument axes_arg   = inputs.at(3)->eval(false);
        if(not starts_arg.empty() and not ends_arg.empty() and not axes_arg.empty())
        {
            std::vector<int64_t> starts_vec;
            std::vector<int64_t> ends_vec;
            std::vector<int64_t> axes_vec;
            starts_arg.visit([&](auto output) { starts_vec.assign(output.begin(), output.end()); });
            ends_arg.visit([&](auto output) { ends_vec.assign(output.begin(), output.end()); });
            axes_arg.visit([&](auto output) { axes_vec.assign(output.begin(), output.end()); });
            m.replace_instruction(
                ins,
                make_op("slice", {{"starts", starts_vec}, {"ends", ends_vec}, {"axes", axes_vec}}),
                inputs.at(0));
        }
    }
};

/**
 * Simplify dimensions_of to a literal when the input arugment has a static shape
 * or the dynamic dimensions from `start` to `end` are fixed.
 */
struct find_static_dimensions_of
{
    auto matcher() const { return match::name("dimensions_of")(); }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins                 = mr.result;
        auto input               = ins->inputs().at(0);
        auto dimensions_of_value = ins->get_operator().to_value();
        auto start               = dimensions_of_value.at("start").to<std::size_t>();
        auto end                 = dimensions_of_value.at("end").to<std::size_t>();
        if(input->get_shape().dynamic())
        {
            // check if dynamic dimensions from start to end are fixed
            auto dds = input->get_shape().dyn_dims();
            if(std::any_of(dds.begin() + start, dds.begin() + end, [](auto dd) {
                   return not dd.is_fixed();
               }))
            {
                return;
            }
        }
        std::size_t output_ndim = end - start;
        std::vector<int64_t> vec_shape(output_ndim);
        migraphx::shape s(migraphx::shape::int64_type, {output_ndim});
        std::vector<std::size_t> input_lens = input->get_shape().to_static(1).lens();
        std::transform(input_lens.begin() + start,
                       input_lens.begin() + end,
                       vec_shape.begin(),
                       [](auto i) { return int64_t(i); });
        migraphx::shape output_shape{migraphx::shape::int64_type, {end - start}};
        auto lit_ins = m.add_literal(migraphx::literal{output_shape, vec_shape});
        m.replace_instruction(ins, lit_ins);
    }
};

/**
 * Simplify allocate into 2 argument reshape that has constant output dimensions into a static 1
 * argument reshape. Intended to simplify what ONNX parse_reshape creates for dynamic reshapes.
 * From:
 * x = allocate(constant_output_dims) -> reshape(data, x)
 * To:
 * reshape(data); reshape.dims = constant_output_dims
 */
struct find_const_alloc_reshapes
{
    auto matcher() const
    {
        return match::name("reshape")(match::nargs(2),
                                      match::arg(1)(match::name("allocate")(match::is_constant())));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto reshape_ins         = mr.result;
        auto reshape_inputs      = reshape_ins->inputs();
        auto alloc_ins           = reshape_inputs.at(1);
        argument output_dims_arg = alloc_ins->inputs().at(0)->eval();
        std::vector<int64_t> output_dims_vec;
        output_dims_arg.visit(
            [&](auto output) { output_dims_vec.assign(output.begin(), output.end()); });
        m.replace_instruction(
            reshape_ins, make_op("reshape", {{"dims", output_dims_vec}}), reshape_inputs.at(0));
        // have dead_code_elimination remove the previous allocate
    }
};

void simplify_dyn_ops::apply(module& m) const
{
    match::find_matches(m,
                        find_static_dimensions_of{},
                        find_const_alloc_reshapes{},
                        find_static_2in_broadcasts{},
                        find_const_3in_slice{},
                        find_const_4in_slice{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
