/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/rewrite_reshapes.hpp>
#include <iterator>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_POINTWISE_FUSION)

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static literal get_scalar(instruction_ref ins)
{
    if(contains({"contiguous", "broadcast", "multibroadcast"}, ins->name()))
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

static shape to_scalar(const shape& s) { return shape{s.type()}; }

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
                    pm->add_parameter(param_name(i), shape{input->get_shape().type()});
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

static module::with_inputs
append_pointwise_module(module_ref parent, instruction_ref ins, instruction_ref output)
{
    assert(contains(output->inputs(), ins));
    module pm     = *ins->module_inputs().at(0);
    module_ref xm = output->module_inputs().at(0);

    assert(pm.get_returns().size() == 1);

    std::unordered_map<instruction_ref, instruction_ref> map_ins =
        pm.get_ins_param_map(ins->inputs());
    map_ins[ins] = pm.get_returns().front();
    auto returns = pm.fuse(*xm, output->inputs(), &map_ins, nullptr, &to_scalar);
    if(ins->outputs().size() > 1)
    {
        auto ireturns = pm.get_returns();
        returns.insert(returns.end(), ireturns.begin(), ireturns.end());
    }
    pm.replace_return(returns);
    auto inputs = find_inputs(map_ins, parent, &pm);
    return {std::move(pm), inputs};
}

static auto find_input_pointwise(instruction_ref ins, bool multi_out)
{
    auto it = std::find_if(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
        return i->name() == "pointwise" and i->outputs().size() == 1;
    });
    if(it == ins->inputs().end() and multi_out)
    {
        it = std::find_if(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
            return i->name() == "pointwise" and
                   std::none_of(i->outputs().begin(), i->outputs().end(), [&](auto output) {
                       return output != ins and reaches(output, ins);
                   });
        });
    }
    return it;
}

static void move_output_instructions_after(module& m, instruction_ref src, instruction_ref dst)
{
    auto d = std::distance(src, dst);
    std::vector<std::pair<std::size_t, instruction_ref>> instructions;
    fix([&](auto self, instruction_ref ins) {
        for(auto output : ins->outputs())
        {
            if(any_of(instructions, [&](const auto& p) { return p.second == output; }))
                continue;
            auto i = std::distance(src, output);
            if(i >= d)
                continue;
            instructions.emplace_back(i, output);
            self(output);
        }
    })(src);
    std::sort(instructions.begin(), instructions.end(), by(std::less<>{}, [](auto&& p) {
                  return p.first;
              }));
    auto loc = std::next(dst);
    for(auto [i, ins] : instructions)
        m.move_instruction(ins, loc);
}

static void
merge_instruction(module_pass_manager& mpm, instruction_ref input, instruction_ref output)
{
    const bool has_multi_out = input->outputs().size() > 1;
    auto fused               = append_pointwise_module(&mpm.get_module(), input, output);
    auto name                = fused.mod.name();
    mpm.rename_module(name, name + ":" + output->module_inputs().front()->name() + "-deleted");
    auto* new_pm = mpm.create_module(name, std::move(fused.mod));
    auto fins =
        mpm.get_module().insert_instruction(output, input->get_operator(), fused.inputs, {new_pm});
    if(has_multi_out)
    {
        auto noutputs = std::max<std::size_t>(1, output->get_shape().sub_shapes().size());
        auto finput   = mpm.get_module().insert_instruction(
            output, make_op("get_tuple_elem", {{"index", noutputs}}), fins);
        move_output_instructions_after(mpm.get_module(), input, finput);
        mpm.get_module().replace_instruction(input, finput);
        if(noutputs == 1)
            fins = mpm.get_module().insert_instruction(
                output, make_op("get_tuple_elem", {{"index", 0}}), fins);
    }
    mpm.get_module().replace_instruction(output, fins);
}

static bool find_pointwise_modules(module_pass_manager& mpm, bool multi_out)
{
    bool changed = false;
    auto last    = std::prev(mpm.get_module().end());
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(ins->name() != "pointwise")
            continue;
        if(ins->outputs().empty() and ins != last)
            continue;
        auto it = find_input_pointwise(ins, multi_out);
        if(it == ins->inputs().end())
            continue;
        auto input = *it;
        merge_instruction(mpm, input, ins);

        changed = true;
    }
    return changed;
}

namespace {
struct pointwise_reshape : rewrite_reshapes_base
{
    static std::string name() { return "pointwise"; }
};

struct pointwise_broadcast_pointwise
{
    auto matcher() const
    {
        auto broadcast_pointwise =
            match::name("multibroadcast")(
                match::used_once(),
                match::args(match::name("pointwise")(match::used_once()).bind("x")))
                .bind("broadcast");
        return match::name("pointwise")(match::any_of[match::inputs()](broadcast_pointwise));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto broadcast_ins = r.instructions["broadcast"];
        auto x_ins         = r.instructions["x"];

        auto broadcast = broadcast_ins->get_operator();

        auto x_inputs = x_ins->inputs();
        std::transform(x_inputs.begin(), x_inputs.end(), x_inputs.begin(), [&](auto input) {
            return m.insert_instruction(broadcast_ins, broadcast, input);
        });

        m.replace_instruction(
            broadcast_ins, x_ins->get_operator(), x_inputs, x_ins->module_inputs());
    }
};

} // namespace

static void rewrite_broadcasts(module_pass_manager& mpm)
{
    match::find_matches(mpm.get_module(), pointwise_broadcast_pointwise{});
    mpm.run_pass(dead_code_elimination{});
}

void fuse_pointwise::apply(module_pass_manager& mpm) const
{
    mpm.run_pass(eliminate_identity{});
    create_pointwise_modules(mpm);
    mpm.run_pass(dead_code_elimination{});
    if(enabled(MIGRAPHX_DISABLE_POINTWISE_FUSION{}))
    {
        return;
    }
    for(int i = 0; i < 8; i++)
    {
        if(enable_rewrite_reshapes)
            mpm.run_pass(rewrite_reshapes<pointwise_reshape>{});
        if(enable_rewrite_broadcasts)
            rewrite_broadcasts(mpm);
        if(not find_pointwise_modules(mpm, enable_multi_output))
            break;
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
