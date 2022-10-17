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

// Handling opt dyn_dims calculation
std::vector<std::size_t> compute_broadcasted_opt_lens(std::vector<std::size_t> s0,
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
            if(a == b)
            {
                return a;
            }
            else if(a == 1 or b == 1)
            {
                return std::max(a, b);
            }
            else
            {
                // if not matching nor 1, set to 0
                return static_cast<std::size_t>(0);
            }
        });
    return out_lens;
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
        auto c_type = compute_common_types(to_shapes(inputs));
        // broadcast all inputs combinations
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto a_input) {
            const auto& ori_input = a_input;
            // multibroadcast this input between every other input
            std::for_each(inputs.cbegin(), inputs.cend(), [&](auto b_input) {
                if(b_input != ori_input)
                {
                    a_input =
                        m.insert_instruction(ins, make_op("multibroadcast"), a_input, b_input);
                }
            });
            if(a_input->get_shape().type() != c_type)
            {
                a_input = m.insert_instruction(
                    ins, make_op("convert", {{"target_type", c_type}}), a_input);
            }
            return a_input;
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
