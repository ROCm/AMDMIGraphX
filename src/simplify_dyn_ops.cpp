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
#include <migraphx/tensor_view.hpp>

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
    auto matcher() const
    {
        return match::name("slice")(match::nargs(2), match::arg(1)(match::is_constant()));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins       = mr.result;
        auto inputs    = ins->inputs();
        auto slice_op  = any_cast<op::slice>(ins->get_operator());
        auto set_attrs = slice_op.get_set_attributes();
        std::vector<int64_t> starts_vec;
        std::vector<int64_t> ends_vec;
        std::vector<int64_t> axes_vec;
        if(set_attrs == op::slice::ends_axes)
        {
            // slice(data, starts)
            inputs.at(1)->eval().visit(
                [&](auto output) { starts_vec.assign(output.begin(), output.end()); });
            ends_vec = slice_op.ends;
            axes_vec = slice_op.axes;
        }
        else if(set_attrs == op::slice::starts_axes)
        {
            // slice(data, ends)
            inputs.at(1)->eval().visit(
                [&](auto output) { ends_vec.assign(output.begin(), output.end()); });
            starts_vec = slice_op.starts;
            axes_vec   = slice_op.axes;
        }
        else
        {
            // slice(data, axes)
            inputs.at(1)->eval().visit(
                [&](auto output) { axes_vec.assign(output.begin(), output.end()); });
            starts_vec = slice_op.starts;
            ends_vec   = slice_op.ends;
        }
        m.replace_instruction(
            ins,
            make_op("slice", {{"starts", starts_vec}, {"ends", ends_vec}, {"axes", axes_vec}}),
            inputs.at(0));
    }
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
        auto slice_op       = any_cast<op::slice>(ins->get_operator());
        auto set_attrs      = slice_op.get_set_attributes();
        std::vector<int64_t> starts_vec;
        std::vector<int64_t> ends_vec;
        std::vector<int64_t> axes_vec;
        if(set_attrs == op::slice::axes_only)
        {
            // slice(data, starts, ends)
            inputs.at(1)->eval().visit(
                [&](auto output) { starts_vec.assign(output.begin(), output.end()); });
            inputs.at(2)->eval().visit(
                [&](auto output) { ends_vec.assign(output.begin(), output.end()); });
            axes_vec = slice_op.axes;
        }
        else if(set_attrs == op::slice::ends_only)
        {
            // slice(data, starts, axes)
            inputs.at(1)->eval().visit(
                [&](auto output) { starts_vec.assign(output.begin(), output.end()); });
            inputs.at(2)->eval().visit(
                [&](auto output) { axes_vec.assign(output.begin(), output.end()); });
            ends_vec = slice_op.ends;
        }
        else
        {
            // slice(data, ends, axes)
            inputs.at(1)->eval().visit(
                [&](auto output) { ends_vec.assign(output.begin(), output.end()); });
            inputs.at(2)->eval().visit(
                [&](auto output) { axes_vec.assign(output.begin(), output.end()); });
            starts_vec = slice_op.starts;
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
 * This matcher can be generalized to matching reshape(data, static_shape_output_tensor).
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
        argument output_dims_arg = alloc_ins->inputs().at(0)->eval(false);
        std::vector<int64_t> output_dims_vec;
        output_dims_arg.visit(
            [&](auto output) { output_dims_vec.assign(output.begin(), output.end()); });
        m.replace_instruction(
            reshape_ins, make_op("reshape", {{"dims", output_dims_vec}}), reshape_inputs.at(0));
        // have dead_code_elimination remove the previous allocate
    }
};

/**
 * Simplify allocate into fill operator that has constant output dimensions and constant value.
 * The allocate into fill instructions is what is produced when parsing the ONNX
 * ConstantOfShape operator. This replacement could be handled with propagate_constant, but
 * would rather have the simplification happen earlier during compiling.
 * This matcher can be generalized to matching fill(constant_value, static_shape_output_tensor).
 * From:
 * x = allocate(constant_ouptut_dims) -> fill(constant_value, x)
 * To:
 * literal
 */
struct find_const_alloc_fill
{
    auto matcher() const
    {
        return match::name("fill")(match::arg(0)(match::is_constant()),
                                   match::arg(1)(match::name("allocate")(match::is_constant())));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto fill_ins = mr.result;
        auto fill_arg = fill_ins->eval(false);
        auto l        = m.add_literal(fill_arg.get_shape(), fill_arg.data());
        m.replace_instruction(fill_ins, l);
    }
};

/**
 * Go through `select_module` instructions and update the `output_dyn_shapes` attribute.
 * Checks the submodule output shapes and determines an appropriate `output_dyn_shapes` attribute.
 * This version ignores dynamic_dimension opt values.
 * Intended to be run after the other simplify_dyn_ops passes.
 */
struct simplify_select_module_output_shape
{
    auto matcher() const { return match::name("select_module"); }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto sm_ins           = mr.result;
        auto sm_module_inputs = sm_ins->module_inputs();
        std::vector<std::vector<shape>> all_output_shapes(sm_module_inputs.size());
        std::transform(sm_module_inputs.begin(),
                       sm_module_inputs.end(),
                       all_output_shapes.begin(),
                       [](auto submod) { return submod->get_output_shapes(); });
        auto shapes_ndim  = get_shapes_ndim(all_output_shapes.front());
        auto shapes_types = get_shapes_types(all_output_shapes.front());
        // check that all of the submodules have the same number of outputs and all respective
        // outputs have the same rank and type
        bool check = std::all_of(
            all_output_shapes.begin() + 1, all_output_shapes.end(), [&](auto out_shapes) {
                bool same_types = get_shapes_types(out_shapes) == shapes_types;
                bool same_ndim  = get_shapes_ndim(out_shapes) == shapes_ndim;
                return same_types and same_ndim;
            });
        if(not check)
        {
            return;
        }
        auto num_out_shapes = shapes_ndim.size();
        std::vector<shape> dyn_shapes{num_out_shapes};
        auto num_submod = sm_module_inputs.size();
        // compare respective output shapes from each submodule to get a range for the output shape
        for(int i : range(num_out_shapes))
        {
            std::vector<shape> shapes_at_index{num_submod};
            std::transform(all_output_shapes.begin(),
                           all_output_shapes.end(),
                           shapes_at_index.begin(),
                           [&](auto output_shapes) { return output_shapes.at(i); });
            dyn_shapes.at(i) = dyn_shape_from_shapes(shapes_at_index);
        }
        auto tuple_shape = shape{dyn_shapes};
        m.replace_instruction(
            sm_ins,
            make_op("select_module", {{"output_dyn_shapes", to_value(tuple_shape)}}),
            sm_ins->inputs(),
            sm_module_inputs);
    }

    std::vector<std::size_t> get_shapes_ndim(std::vector<shape> shapes) const
    {
        std::vector<std::size_t> ret{shapes.size()};
        std::transform(
            shapes.cbegin(), shapes.cend(), ret.begin(), [](auto s) { return s.ndim(); });
        return ret;
    }

    std::vector<shape::type_t> get_shapes_types(std::vector<shape> shapes) const
    {
        std::vector<shape::type_t> ret{shapes.size()};
        std::transform(
            shapes.cbegin(), shapes.cend(), ret.begin(), [](auto s) { return s.type(); });
        return ret;
    }

    /**
     * Equivalent to creating a 2D matrix of shape lengths and do a reduce_min over each axis.
     * The shapes can be dynamic or static.
     * Assuming all shapes have the same ndim.
     */
    shape dyn_shape_from_shapes(std::vector<shape> shapes) const
    {
        // making 2D matrices of min_lens and max_lens
        std::vector<std::size_t> all_min_lens;
        std::vector<std::size_t> all_max_lens;
        for(int i : range(shapes.size()))
        {
            auto s        = shapes.at(i);
            auto min_lens = s.min_lens();
            auto max_lens = s.max_lens();
            for(auto l : min_lens)
            {
                all_min_lens.push_back(l);
            }
            for(auto l : max_lens)
            {
                all_max_lens.push_back(l);
            }
        }
        assert(all_min_lens.size() == shapes.size() * shapes.front().ndim());
        assert(all_max_lens.size() == shapes.size() * shapes.front().ndim());
        auto num_rows = shapes.size();
        auto num_cols = shapes.front().ndim();
        shape tensor_shape{shapes.front().type(), {num_rows, num_cols}};
        auto min_lens_matrix = make_view(tensor_shape, all_min_lens.data());
        auto max_lens_matrix = make_view(tensor_shape, all_max_lens.data());

        std::vector<std::size_t> mins;
        std::vector<std::size_t> maxes;
        // rearranging data into column vectors to reduce over
        // i = row, j = column
        for(int j : range(num_cols))
        {
            std::vector<std::size_t> reduce_min_vals{num_rows};
            std::vector<std::size_t> reduce_max_vals{num_rows};
            for(int i : range(num_rows))
            {
                reduce_min_vals.at(i) = min_lens_matrix(i, j);
                reduce_max_vals.at(i) = min_lens_matrix(i, j);
            }
            std::size_t max_int = std::numeric_limits<std::size_t>::max();
            std::size_t min_val =
                std::reduce(reduce_min_vals.begin(),
                            reduce_min_vals.end(),
                            max_int,
                            [](std::size_t x, std::size_t y) { return x < y ? x : y; });
            std::size_t max_val =
                std::reduce(reduce_max_vals.begin(),
                            reduce_max_vals.end(),
                            0,
                            [](std::size_t x, std::size_t y) { return x > y ? x : y; });
            mins.push_back(min_val);
            maxes.push_back(max_val);
        }
        // fixed output shape case
        if(mins == maxes)
        {
            return shape{shapes.front().type(), mins};
        }
        // dynamic output shape case
        return shape{shapes.front().type(), mins, maxes, {}};
    }
};

void simplify_dyn_ops::apply(module& m) const
{
    match::find_matches(m,
                        find_static_dimensions_of{},
                        find_const_alloc_reshapes{},
                        find_static_2in_broadcasts{},
                        find_const_2in_slice{},
                        find_const_3in_slice{},
                        find_const_4in_slice{},
                        find_const_alloc_fill{});
    match::find_matches(m, simplify_select_module_output_shape{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
