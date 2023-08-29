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
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Convert 2 input static shape broadcast/multibroadcast into 1 input version.
 * Some compiler passes (ex. simplify_algebra) only support the 1 input versions
 * of the broadcasting operators.
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
 * Simplify slice with variable `starts` and `ends` to the constant version if
 * the `starts` and `ends` inputs are constant.
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
        auto starts_in      = ins->inputs().at(1);
        auto ends_in        = ins->inputs().at(2);
        argument starts_arg = starts_in->eval();
        argument ends_arg   = ends_in->eval();
        if(not(starts_arg.empty() and ends_arg.empty()))
        {
            m.replace_instrcution(
                    ins,
                    make_op("slice", {
                {
                    "starts",
                },
                    {
                        "ends",
                    },
                {
                    "axes",
                }
        }
        }
    };

    /**
     * Simplify slice with variable `starts`, `ends`, and `input_axes` to the constant version if
     * the `starts` and `ends` inputs are constant.
     */
    struct find_const_4in_slice
    {
        auto matcher() const
        {
            return match::name("slice")(match::nargs(3),
                                        match::arg(1)(match::is_constant()),
                                        match::arg(2)(match::is_constant()),
                                        match::arg(3)(match::is_constant()));
        }

        void apply(module& m, const match::matcher_result& mr) const {}
    };

    void simplify_dyn_ops::apply(module& m) const
    {
        match::find_matches(
            m, find_static_2in_broadcasts{}, find_const_3in_slice{}, find_const_4in_slice{});
    }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
