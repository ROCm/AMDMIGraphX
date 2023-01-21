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
        check_shapes{inputs, *this}.has(sm->get_parameter_shapes().size()).same_dims();
        auto s    = inputs.at(0);
        auto lens = s.lens();
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

        // TODO: Ensure standard shape
        auto x0 = rm->add_parameter("x0", ins->inputs().front()->get_shape());
        auto r  = rm->add_instruction(ins->get_operator(), x0);
        rm->add_return({r});

        auto v = ins->get_operator().to_value();
        mpm.get_module().replace_instruction(
            ins, make_op("fused_reduce", {{"axes", v["axes"]}}), ins->inputs(), {rm});
    }
}

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

static std::vector<instruction_ref> get_returns(module& m)
{
    auto last = std::prev(m.end());
    if(last->name() == "@return")
        return last->inputs();
    return {last};
}

struct find_reduce_pointwise
{
    auto matcher() const
    {
        return match::name("pointwise")(match::any_of[match::inputs()](
            match::name("fused_reduce")(match::used_once()).bind("reduce")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto reduce = r.instructions["reduce"];

        const auto* old_rm = reduce->module_inputs().front();
        auto* rm           = mpm.create_module(old_rm->name() + ":pointwise");
        rm->set_bypass();
        // Copy module instructions
        rm->add_instructions(old_rm);
        auto map_ins    = get_ins_param_map(reduce->inputs(), rm);
        auto new_inputs = reduce->inputs();
        for(auto input : ins->inputs())
        {
            if(contains(map_ins, input))
                continue;
            if(input == reduce)
            {
                map_ins[input] = get_returns(*rm).front();
            }
            else
            {
                map_ins[input] =
                    rm->add_parameter("x" + std::to_string(new_inputs.size()), input->get_shape());
                new_inputs.push_back(input);
            }
        }

        auto out = rm->add_instructions({ins}, map_ins);
        rm->add_return(out);
        mpm.get_module().replace_instruction(ins, reduce->get_operator(), new_inputs, {rm});
    }
};

void fuse_reduce::apply(module_pass_manager& mpm) const
{
    create_reduce_modules(mpm);
    mpm.run_pass(dead_code_elimination{});
    match::find_matches(mpm, find_reduce_pointwise{});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
