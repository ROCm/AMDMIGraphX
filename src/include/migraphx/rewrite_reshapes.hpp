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
 *
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHX_REWRITE_RESHAPES_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_REWRITE_RESHAPES_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/common_dims.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/dead_code_elimination.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct rewrite_reshapes_base
{
    template <class AxesMap>
    static instruction_ref insert(module_pass_manager& mpm,
                                  instruction_ref ins,
                                  const std::vector<instruction_ref>& inputs,
                                  const AxesMap&)
    {
        return mpm.get_module().insert_instruction(
            ins, ins->get_operator(), inputs, ins->module_inputs());
    }

    template <class AxesMap>
    static bool supports(instruction_ref, std::vector<std::size_t>&, const AxesMap&)
    {
        return true;
    }

    static std::vector<std::size_t> base_dims(instruction_ref ins)
    {
        return ins->get_shape().lens();
    }
};

template <class T>
struct rewrite_reshapes
{
    std::string name() const { return "rewrite_reshapes"; }
    struct find_op_reshape_op
    {
        std::string op1;
        std::string op2;

        auto matcher() const
        {
            auto reshape =
                match::name("reshape", "squeeze", "unsqueeze", "flatten")(match::used_once());
            auto skip_contiguous_broadcast =
                match::skip(match::name("contiguous", "multibroadcast")(match::used_once()));
            auto skip_contiguous_broadcast_arg = [&](auto... ms) {
                return match::arg(0)(skip_contiguous_broadcast(ms...));
            };
            auto pointwise         = match::name(op1)(match::used_once());
            auto reshape_pointwise =
                reshape(skip_contiguous_broadcast_arg(pointwise.bind("x"))).bind("reshape");
            return match::name(op2)(match::any_of[match::inputs()](
                skip_contiguous_broadcast(reshape_pointwise).bind("input")));
        }

        template <class F>
        static instruction_ref find_input_if(instruction_ref start, instruction_ref last, F f)
        {
            while(start != last)
            {
                if(f(start))
                    return start;
                if(start->inputs().size() != 1)
                    return last;
                start = start->inputs().front();
            }
            return last;
        }

        static bool match_input(instruction_ref ins, instruction_ref x_ins)
        {
            if(ins->inputs().empty())
                return false;
            auto input = ins->inputs().front();
            if(input->name() == "contiguous")
                return match_input(input, x_ins);
            return x_ins == input;
        }

        static std::optional<bool> is_broadcasted(instruction_ref start, instruction_ref last)
        {
            auto broadcast_ins =
                find_input_if(start, last, [&](auto i) { return i->name() == "multibroadcast"; });
            bool result = broadcast_ins != last;
            if(result and not match_input(broadcast_ins, last))
                return nullopt;
            return result;
        }

        void apply(module_pass_manager& mpm, const match::matcher_result& r) const
        {
            auto ins         = r.result;
            auto x_ins       = r.instructions["x"];
            auto reshape_ins = r.instructions["reshape"];
            auto input_ins   = r.instructions["input"];

            const auto has_broadcast_before_reshape = is_broadcasted(reshape_ins, x_ins);
            const auto has_broadcast_after_reshape  = is_broadcasted(input_ins, reshape_ins);
            if(not has_broadcast_before_reshape.has_value())
                return;
            if(not has_broadcast_after_reshape.has_value())
                return;
            if(*has_broadcast_after_reshape and *has_broadcast_before_reshape)
                return;
            const bool has_broadcast =
                *has_broadcast_after_reshape or *has_broadcast_before_reshape;

            auto dims1 = T::base_dims(ins);
            auto dims2 = T::base_dims(x_ins);

            if(elements(dims1) != elements(dims2))
                return;

            auto cd = common_dims::compute(T::base_dims(ins), T::base_dims(x_ins));
            if(cd.dims.empty())
                return;

            if(ins->name() != "pointwise" and not T::supports(ins, cd.dims, cd.axes_map1))
                return;
            if(x_ins->name() != "pointwise" and not T::supports(x_ins, cd.dims, cd.axes_map2))
                return;

            auto reshape_input = [&](const auto& ins_to_insert) {
                return [&](auto input) {
                    auto dims = cd.get_dimensions_for(input->get_shape().lens());
                    return mpm.get_module().insert_instruction(
                        ins_to_insert, make_op("reshape", {{"dims", dims}}), input);
                };
            };
            auto x_inputs = x_ins->inputs();
            std::transform(
                x_inputs.begin(), x_inputs.end(), x_inputs.begin(), reshape_input(x_ins));
            auto new_x_ins = insert(mpm, x_ins, x_inputs, cd.axes_map2);
            if(has_broadcast)
            {
                new_x_ins = mpm.get_module().insert_instruction(
                    x_ins, make_op("multibroadcast", {{"out_lens", cd.dims}}), new_x_ins);
            }

            auto inputs = ins->inputs();
            std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
                if(input == input_ins)
                    return new_x_ins;
                return reshape_input(ins)(input);
            });
            auto pw = insert(mpm, ins, inputs, cd.axes_map1);
            mpm.get_module().replace_instruction(
                ins, make_op("reshape", {{"dims", ins->get_shape().lens()}}), pw);
        }

        static bool same_dims(instruction_ref ins)
        {
            return all_of(ins->inputs(), [&](auto input) {
                return input->get_shape().lens() == ins->get_shape().lens();
            });
        }

        template <class AxesMap>
        static instruction_ref insert(module_pass_manager& mpm,
                                      instruction_ref ins,
                                      const std::vector<instruction_ref>& inputs,
                                      const AxesMap& am)
        {
            if(ins->name() == "pointwise")
                return mpm.get_module().insert_instruction(
                    ins, ins->get_operator(), inputs, ins->module_inputs());
            return T::insert(mpm, ins, inputs, am);
        }
    };

    void apply(module_pass_manager& mpm) const
    {
        if(T::name() == "pointwise")
        {
            match::find_matches(mpm, find_op_reshape_op{"pointwise", T::name()});
        }
        else
        {
            match::find_matches(mpm,
                                find_op_reshape_op{"pointwise", T::name()},
                                find_op_reshape_op{T::name(), "pointwise"},
                                find_op_reshape_op{T::name(), T::name()});
        }
        mpm.run_pass(simplify_reshapes{1});
        mpm.run_pass(eliminate_common_subexpression{});
        mpm.run_pass(dead_code_elimination{});
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_REWRITE_RESHAPES_HPP
