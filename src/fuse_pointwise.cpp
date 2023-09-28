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
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/common_dims.hpp>
#include <iterator>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_POINTWISE_FUSION)

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static literal get_scalar(instruction_ref ins)
{
    if(ins->name() == "contiguous")
        return get_scalar(ins->inputs().front());
    const auto& s = ins->get_shape();
    if(s.elements() != 1 and not(s.scalar()))
        return {};
    if(not ins->can_eval())
        return {};
    auto e = ins->eval();
    literal r{};
    // needed for bool as visit_at invokes as() which promotes bool to int8
    // Without this we'll break type checks for logical ops that are fused.
    if(e.get_shape().type() == shape::bool_type)
    {
        r = literal{e.at<bool>()};
    }
    else
    {
        e.visit_at([&](auto x) { r = literal{x}; });
    }
    return r;
}

static void create_pointwise_modules(module_pass_manager& mpm)
{
    std::size_t n = 0;
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(not ins->get_operator().attributes().get("pointwise", false))
            continue;
        if(ins->get_operator().name() == "layout")
            continue;
        auto* pm = mpm.create_module(mpm.get_module().name() + ":pointwise" + std::to_string(n++));
        pm->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> param_map;
        std::vector<instruction_ref> pointwise_inputs;
        std::size_t i = 0;

        for(auto input : ins->inputs())
        {
            if(contains(param_map, input))
                continue;
            auto scalar = get_scalar(input);
            if(scalar.empty())
            {
                pointwise_inputs.push_back(input);
                param_map[input] =
                    pm->add_parameter("x" + std::to_string(i), shape{input->get_shape().type()});
                i++;
            }
            else
            {
                param_map[input] = pm->add_literal(scalar);
            }
        }

        // Don't create pointwise module if no inputs are detected
        if(pointwise_inputs.empty())
            continue;

        std::vector<instruction_ref> inputs;
        std::transform(ins->inputs().begin(),
                       ins->inputs().end(),
                       std::back_inserter(inputs),
                       [&](auto input) { return param_map[input]; });
        auto r = pm->add_instruction(ins->get_operator(), inputs);
        pm->add_return({r});

        mpm.get_module().replace_instruction(ins, make_op("pointwise"), pointwise_inputs, {pm});
    }
}

static std::vector<instruction_ref> append_pointwise_module(instruction_ref ins,
                                                            instruction_ref output)
{
    assert(contains(output->inputs(), ins));
    module_ref pm = ins->module_inputs().at(0);
    module_ref xm = output->module_inputs().at(0);

    auto last = std::prev(pm->end());
    assert(last->name() == "@return");
    assert(last->inputs().size() == 1);

    assert(pm->get_parameter_names().size() == ins->inputs().size());
    assert(xm->get_parameter_names().size() == output->inputs().size());

    std::vector<instruction_ref> inputs = ins->inputs();
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    std::unordered_map<instruction_ref, instruction_ref> input_map;
    // Copy inputs to input_map
    for(auto i : range(inputs.size()))
    {
        auto input = inputs[i];
        auto param = pm->get_parameter("x" + std::to_string(i));
        assert(param != pm->end());
        input_map[input] = param;
    }
    // Add the new parameter and additional inputs
    for(auto i : range(output->inputs().size()))
    {
        auto input = output->inputs()[i];
        auto param = xm->get_parameter("x" + std::to_string(i));
        assert(param != xm->end());
        if(input == ins)
        {
            map_ins[param]   = last->inputs().front();
            input_map[input] = map_ins[param];
        }
        // Avoid duplicate paramter inputs
        else if(contains(input_map, input))
        {
            map_ins[param] = input_map[input];
        }
        else
        {
            map_ins[param] =
                pm->add_parameter("x" + std::to_string(inputs.size()), {input->get_shape().type()});
            inputs.push_back(input);
            input_map[input] = map_ins[param];
        }
    }
    pm->replace_return(pm->insert_instructions(last, xm, map_ins));
    return inputs;
}

static bool find_pointwise_modules(module& m)
{
    bool changed = false;
    auto last    = std::prev(m.end());
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "pointwise")
            continue;
        if(ins->outputs().empty() and ins != last)
            continue;
        auto it = std::find_if(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
            return i->name() == "pointwise" and i->outputs().size() == 1;
        });
        if(it == ins->inputs().end())
            continue;
        auto input = *it;

        auto new_inputs = append_pointwise_module(input, ins);
        m.replace_instruction(input, input->get_operator(), new_inputs, input->module_inputs());
        m.replace_instruction(ins, input);
        m.move_instruction(input, ins);

        changed = true;
    }
    return changed;
}
namespace {
struct find_pointwise_reshape_pointwise
{
    auto matcher() const
    {
        auto reshape =
            match::name("reshape", "squeeze", "unsqueeze", "flatten")(match::used_once());
        auto skip_contiguous = [](auto... ms) {
            return match::arg(0)(match::skip(match::name("contiguous")(match::used_once()))(ms...));
        };
        auto pointwise         = match::name("pointwise")(match::used_once());
        auto reshape_pointwise = reshape(skip_contiguous(pointwise.bind("x"))).bind("reshape");
        return match::name("pointwise")(match::any_of[match::inputs()](reshape_pointwise));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins         = r.result;
        auto x_ins       = r.instructions["x"];
        auto reshape_ins = r.instructions["reshape"];

        auto cd = common_dims::compute(ins->get_shape().lens(), x_ins->get_shape().lens());
        if(cd.dims.empty())
            return;

        auto reshape_input = [&](const auto& ins_to_insert) {
            return [&](auto input) {
                auto c = m.insert_instruction(ins_to_insert, make_op("contiguous"), input);
                return m.insert_instruction(
                    ins_to_insert, make_op("reshape", {{"dims", cd.dims}}), c);
            };
        };
        auto x_inputs = x_ins->inputs();
        std::transform(x_inputs.begin(), x_inputs.end(), x_inputs.begin(), reshape_input(x_ins));
        auto new_x_ins =
            m.insert_instruction(x_ins, x_ins->get_operator(), x_inputs, x_ins->module_inputs());

        auto inputs = ins->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            if(input == reshape_ins)
                return new_x_ins;
            return reshape_input(ins)(input);
        });
        auto pw = m.insert_instruction(ins, ins->get_operator(), inputs, ins->module_inputs());
        m.replace_instruction(ins, make_op("reshape", {{"dims", ins->get_shape().lens()}}), pw);
    }
};
} // namespace

void fuse_pointwise::apply(module_pass_manager& mpm) const
{
    create_pointwise_modules(mpm);
    mpm.run_pass(dead_code_elimination{});
    if(enabled(MIGRAPHX_DISABLE_POINTWISE_FUSION{}))
    {
        return;
    }
    for(int i = 0; i < 8; i++)
    {
        match::find_matches(mpm.get_module(), find_pointwise_reshape_pointwise{});
        mpm.run_pass(simplify_reshapes{1});
        if(not find_pointwise_modules(mpm.get_module()))
            break;
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
