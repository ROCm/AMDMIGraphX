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
#include <migraphx/common.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Example:
// s0 = (3,2,4,5) and s1 = (2,1,1)
//
// In this case we need to broadcast (:,1,1) portion of
// s1 plus broadcast the 1st dimension of s1
// giving output_lens = (3,2,4,5)
//
// Another example:
// s0 = (3,2,1,5) and s1 = (2,7,5)
// In this case we need to broadcast the (:,:,1:,:) axis
// of s0 plus the 1st dimension of s1 giving
// output_lens = (3,2,7,5)
//
std::vector<std::size_t> compute_broadcasted_lens(std::vector<std::size_t> s0,
                                                  std::vector<std::size_t> s1)
{
    if(s0 == s1)
        return s0;
    if(s0.size() > s1.size())
        s0.swap(s1);
    std::vector<std::size_t> out_lens(s1);
    auto offset = s1.size() - s0.size();
    std::transform(
        s0.begin(), s0.end(), s1.begin() + offset, out_lens.begin() + offset, [&](auto a, auto b) {
            if(a != b and a != 1 and b != 1)
            {
                MIGRAPHX_THROW("COMPUTE_BROADCASTLEN: shape {" + migraphx::to_string_range(s0) +
                               "} and {" + migraphx::to_string_range(s1) + "} mismatch!");
            }
            return std::max(a, b);
        });
    return out_lens;
}

std::vector<shape::dynamic_dimension> compute_broadcasted_dyn_dims(shape s0, shape s1)
{
    assert(s0.dynamic() or s1.dynamic());
    // change both shapes to dynamic_dimension representation
    if(not s0.dynamic())
        s0 = s0.to_dynamic();
    if(not s1.dynamic())
        s1 = s1.to_dynamic();

    if(s0.ndim() > s1.ndim())
    {
        std::swap(s0, s1);
    }
    auto offset = s1.ndim() - s0.ndim();
    std::vector<shape::dynamic_dimension> out_dims(s1.dyn_dims());
    shape::dynamic_dimension one_dyn_dim{1, 1, 0};
    std::transform(
        s0.dyn_dims().cbegin(),
        s0.dyn_dims().cend(),
        s1.dyn_dims().cbegin() + offset,
        out_dims.begin() + offset,
        [&](auto a, auto b) {
            if(a == b)
            {
                return a;
            }
            else if(a == one_dyn_dim or b == one_dyn_dim)
            {
                // setting opt to 0, may need to be changed
                return shape::dynamic_dimension{std::max(a.min, b.min), std::max(a.max, b.max), 0};
            }
            else
            {
                MIGRAPHX_THROW("COMPUTE_BROADCASTED_DYN_DIMS: dynamic shapes {" +
                               migraphx::to_string_range(s0.dyn_dims()) + "} and {" +
                               migraphx::to_string_range(s1.dyn_dims()) + "} mismatch!");
            }
        });
    return out_dims;
}

// Compute the common (broadcasted) dimensions of a list of fixed shapes
std::vector<std::size_t> compute_common_lens(const std::vector<shape>& shapes)
{
    assert(not shapes.empty());
    assert(
        std::none_of(shapes.cbegin(), shapes.cend(), [](auto shape) { return shape.dynamic(); }));
    return transform_accumulate(shapes.begin() + 1,
                                shapes.end(),
                                shapes.front().lens(),
                                &compute_broadcasted_lens,
                                [](auto s) { return s.lens(); });
}

shape::type_t compute_common_type(shape::type_t t1, shape::type_t t2)
{
    if(t1 == t2)
        return t1;
    shape::type_t result;
    shape::visit(t1, [&](auto x) {
        shape::visit(t2, [&](auto y) {
            // Workaround broken warning on gcc 5
            (void)x;
            (void)y;
            using type = std::common_type_t<decltype(x()), decltype(y())>;
            result     = shape::get_type<type>{};
        });
    });
    return result;
}

shape::type_t compute_common_types(const std::vector<shape>& shapes)
{
    assert(not shapes.empty());
    return transform_accumulate(
        shapes.begin() + 1, shapes.end(), shapes.front().type(), &compute_common_type, [&](auto s) {
            return s.type();
        });
}

shape common_shape(const std::vector<shape>& shapes)
{
    if(shapes.empty())
        return {};
    return {compute_common_types(shapes), compute_common_lens(shapes)};
}

instruction_ref insert_common_op(module& m,
                                 instruction_ref ins,
                                 const operation& op,
                                 std::vector<instruction_ref> inputs)
{
    if(std::any_of(
           inputs.cbegin(), inputs.cend(), [](auto input) { return input->get_shape().dynamic(); }))
    {
        // currently only handles the binary case
        if(inputs.size() != 2)
        {
            MIGRAPHX_THROW("INSERT_COMMON_OP: not handled; " + migraphx::to_string(inputs.size()) +
                           "inputs, only handle two inputs if any are dynamic shape");
        }

        auto c_type = compute_common_types(to_shapes(inputs));
        auto c_dyn_dims =
            compute_broadcasted_dyn_dims(inputs[0]->get_shape(), inputs[1]->get_shape());

        // following should work for a static or dynamic shape
        if(inputs[0]->get_shape().dyn_dims() != c_dyn_dims)
        {
            inputs[0] = m.insert_instruction(
                ins,
                make_op("multibroadcast", {{"out_dyn_dims", to_value(c_dyn_dims)}}),
                inputs[0],
                inputs[1]);
        }
        if(inputs[1]->get_shape().dyn_dims() != c_dyn_dims)
        {
            inputs[1] = m.insert_instruction(
                ins,
                make_op("multibroadcast", {{"out_dyn_dims", to_value(c_dyn_dims)}}),
                inputs[1],
                inputs[0]);
        }
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            if(input->get_shape().type() != c_type)
            {
                input =
                    m.insert_instruction(ins, make_op("convert", {{"target_type", c_type}}), input);
            }
            return input;
        });
    }
    else
    {
        auto common = common_shape(to_shapes(inputs));
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            if(input->get_shape().lens() != common.lens())
            {
                input = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"out_lens", common.lens()}}), input);
            }
            if(input->get_shape().type() != common.type())
            {
                input = m.insert_instruction(
                    ins, make_op("convert", {{"target_type", common.type()}}), input);
            }
            return input;
        });
    }
    return m.insert_instruction(ins, op, inputs);
}

instruction_ref add_common_op(module& m, const operation& op, std::vector<instruction_ref> inputs)
{
    return insert_common_op(m, m.end(), op, std::move(inputs));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
