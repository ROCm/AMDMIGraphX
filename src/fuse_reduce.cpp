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
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/register_op.hpp>
#include <iterator>
#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct fused_reduce
{
    std::vector<std::int64_t> axes{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        auto* sm = mods.front();
        if(sm->get_output_shapes().size() != 1)
            MIGRAPHX_THROW("Only one output supported");
        auto names = sm->get_parameter_names();
        check_shapes{inputs, *this}.has(names.size()).same_ndims();
        std::sort(names.begin(), names.end());
        auto shapes = sm->get_parameter_shapes();
        // Check dimension matches for each input
        if(not equal(names, inputs, [&](const auto& name, const auto& input) {
               return shapes.at(name).lens() == input.lens();
           }))
            MIGRAPHX_THROW("Dimenstion does not match the submodule.");
        const auto& s = inputs.at(0);
        auto lens     = s.lens();
        if(lens != sm->get_output_shapes().front().lens())
        {
            for(const auto& axis : axes)
            {
                lens[axis] = 1;
            }
        }

        return shape::from_permutation(
            sm->get_output_shapes().front().type(), lens, find_permutation(inputs));
    }

    std::string name() const { return "fused_reduce"; }
};
MIGRAPHX_REGISTER_OP(fused_reduce);

static std::unordered_map<instruction_ref, instruction_ref>
get_ins_param_map(const std::vector<instruction_ref>& inputs, const_module_ref sm)
{
    std::unordered_map<instruction_ref, instruction_ref> result;
    auto names = sm->get_parameter_names();
    std::sort(names.begin(), names.end());
    assert(names.size() == inputs.size());
    std::transform(names.begin(),
                   names.end(),
                   inputs.begin(),
                   std::inserter(result, result.end()),
                   [&](const auto& name, auto input) {
                       return std::make_pair(input, sm->get_parameter(name));
                   });
    return result;
}

static void insert_params(module_ref sm,
                          instruction_ref ins,
                          std::unordered_map<instruction_ref, instruction_ref>& map_ins)
{
    auto n = sm->get_parameter_shapes().size();
    for(auto input : ins->inputs())
    {
        if(contains(map_ins, input))
            continue;
        auto s         = shape{input->get_shape().type(), input->get_shape().lens()};
        map_ins[input] = sm->add_parameter("x" + std::to_string(n++), s);
    }
}

static auto insert_ins_in_submodule(module_ref sm,
                                    instruction_ref ins,
                                    std::unordered_map<instruction_ref, instruction_ref>& map_ins)
{
    insert_params(sm, ins, map_ins);
    return sm->add_instructions({ins}, map_ins);
}

static auto insert_ins_in_submodule(module_ref sm, instruction_ref ins)
{
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    return insert_ins_in_submodule(sm, ins, map_ins);
}

static auto
insert_module_in_submodule(module_ref sm,
                           instruction_ref ins,
                           std::unordered_map<instruction_ref, instruction_ref>& map_ins)
{
    insert_params(sm, ins, map_ins);
    auto* m        = ins->module_inputs().front();
    auto param_map = get_ins_param_map(ins->inputs(), m);
    for(auto&& [input, param] : param_map)
    {
        map_ins[param] = map_ins.at(input);
    }
    return sm->add_instructions(m, map_ins);
}

static std::vector<instruction_ref>
find_inputs(module_ref sm,
            const module& parent,
            const std::unordered_map<instruction_ref, instruction_ref>& map_ins)
{
    std::vector<instruction_ref> result;
    std::map<std::string, instruction_ref> names;
    for(auto&& [input, param] : map_ins)
    {
        if(not sm->has_instruction(param))
            continue;
        if(param->name() != "@param")
            continue;
        if(not parent.has_instruction(input))
            continue;
        auto v      = param->get_operator().to_value();
        auto name   = v.at("parameter").to<std::string>();
        names[name] = input;
    }
    std::transform(names.begin(), names.end(), std::back_inserter(result), [](const auto& p) {
        return p.second;
    });
    assert(result.size() == sm->get_parameter_shapes().size());
    return result;
}

static void create_reduce_modules(module_pass_manager& mpm)
{
    std::size_t n = 0;
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(not ins->get_operator().attributes().get("reduce", false))
            continue;
        if(ins->inputs().size() != 1)
            continue;

        auto* rm =
            mpm.create_module(mpm.get_module().name() + ":" + ins->name() + std::to_string(n++));
        rm->set_bypass();

        rm->add_return(insert_ins_in_submodule(rm, ins));
        auto v = ins->get_operator().to_value();
        mpm.get_module().replace_instruction(
            ins, make_op("fused_reduce", {{"axes", v["axes"]}}), ins->inputs(), {rm});
    }
}

template <class... Ms>
static auto match_broadcast(Ms... ms)
{
    return match::skip(match::name("contiguous"))(
        match::name("multibroadcast")(match::arg(0)(ms...), match::used_once()).bind("broadcast"));
}

template <class... Ms>
static auto any_input(Ms... ms)
{
    return match::any_of[match::inputs()](match::any(ms...).bind("input"));
}

static auto match_broadcastable_input(const std::string& op, const std::string& name)
{
    auto match_op                 = match::name(op)(match::used_once()).bind(name);
    auto match_op_input           = any_input(match_op, match::used_once());
    auto broadcast_match_op_input = any_input(match_broadcast(match_op), match::used_once());
    return match::any_of(match_op_input, broadcast_match_op_input);
}

namespace {
struct find_pointwise_reduce
{
    auto matcher() const
    {
        return match::name("fused_reduce")(match_broadcastable_input("pointwise", "pointwise"));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto reduce = r.result;
        auto input  = r.instructions["pointwise"];

        const auto* pm     = input->module_inputs().front();
        const auto* old_rm = reduce->module_inputs().front();
        auto* rm           = mpm.create_module(pm->name() + ":" + old_rm->name());
        rm->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        // Insert pointwise
        auto rins      = insert_ins_in_submodule(rm, input, map_ins).front();
        map_ins[input] = rins;

        if(contains(r.instructions, "broadcast"))
        {
            auto broadcast     = r.instructions["broadcast"];
            map_ins[broadcast] = insert_ins_in_submodule(rm, broadcast, map_ins).front();
        }

        // Insert fused_reduce
        rm->add_return(insert_module_in_submodule(rm, reduce, map_ins));

        auto new_inputs = find_inputs(rm, mpm.get_module(), map_ins);
        mpm.get_module().replace_instruction(reduce, reduce->get_operator(), new_inputs, {rm});
    }
};

struct find_reduce_pointwise
{

    auto matcher() const
    {
        return match::name("pointwise")(match_broadcastable_input("fused_reduce", "reduce"));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto pw     = r.result;
        auto reduce = r.instructions["reduce"];
        auto input  = r.instructions["input"];

        const auto* pm     = pw->module_inputs().front();
        const auto* old_rm = reduce->module_inputs().front();
        auto* rm           = mpm.create_module(old_rm->name() + ":" + pm->name());
        rm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        // Copy module instructions
        insert_module_in_submodule(rm, reduce, map_ins);
        if(contains(r.instructions, "broadcast"))
        {
            auto broadcast                       = r.instructions["broadcast"];
            map_ins[broadcast->inputs().front()] = rm->get_returns().front();
            auto bout                            = insert_ins_in_submodule(rm, broadcast, map_ins);
            map_ins[input]                       = bout.front();
        }
        else
        {
            map_ins[input] = rm->get_returns().front();
        }

        auto out = insert_ins_in_submodule(rm, pw, map_ins);
        rm->replace_return(out);

        auto new_inputs = find_inputs(rm, mpm.get_module(), map_ins);
        mpm.get_module().replace_instruction(pw, reduce->get_operator(), new_inputs, {rm});
    }
};

struct find_reduce_reduce
{
    auto matcher() const
    {
        return match::name("fused_reduce")(match_broadcastable_input("fused_reduce", "reduce"));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto reduce1 = r.result;
        auto reduce2 = r.instructions["reduce"];
        auto input   = r.instructions["input"];

        if(reduce1->get_operator() != reduce2->get_operator())
            return;

        const auto* rm1 = reduce1->module_inputs().front();
        const auto* rm2 = reduce2->module_inputs().front();
        auto* rm        = mpm.create_module(rm1->name() + ":" + rm2->name());
        rm->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        // Copy reduce1 instructions
        insert_module_in_submodule(rm, reduce2, map_ins);
        if(contains(r.instructions, "broadcast"))
        {
            auto broadcast                       = r.instructions["broadcast"];
            map_ins[broadcast->inputs().front()] = rm->get_returns().front();
            auto bout                            = insert_ins_in_submodule(rm, broadcast, map_ins);
            map_ins[input]                       = bout.front();
        }
        else
        {
            map_ins[input] = rm->get_returns().front();
        }

        auto out = insert_module_in_submodule(rm, reduce1, map_ins);
        rm->replace_return(out);

        auto new_inputs = find_inputs(rm, mpm.get_module(), map_ins);
        mpm.get_module().replace_instruction(reduce1, reduce1->get_operator(), new_inputs, {rm});
    }
};

} // namespace

void fuse_reduce::apply(module_pass_manager& mpm) const
{
    create_reduce_modules(mpm);
    mpm.run_pass(dead_code_elimination{});
    for(int i = 0; i < 4; i++)
    {
        match::find_matches(
            mpm, find_reduce_pointwise{}, find_pointwise_reduce{}, find_reduce_reduce{});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
