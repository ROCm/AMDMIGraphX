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
#include <migraphx/split_reduce.hpp>
#include <migraphx/dom_info.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/liveness.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/param_utils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct split_fused_reduce
{
    std::vector<std::int64_t> axes{};
    std::string assign = "assign_none";

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.assign, "assign"));
    }

    value attributes() const { return {{"prefill", 0}}; }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        const auto* sm = mods.front();
        auto names = sm->get_parameter_names();
        check_shapes{inputs, *this}.has(names.size()).same_ndims();

        auto result =
            sm->compute_shapes(inputs, {.name = name(), .strict_type = true, .strict_lens = true});
        if(result.size() == 1)
            return result.front();
        return shape{result};
    }

    std::string name() const { return "split_fused_reduce"; }
};
MIGRAPHX_REGISTER_OP(split_fused_reduce);

static bool is_reduce(const instruction& ins) { return contains(ins.name(), "reduce"); }

static std::vector<instruction_ref> find_split(const_module_ref rm)
{
    std::vector<instruction_ref> result;
    copy_if(
        iterator_for(*rm), std::back_inserter(result), [](auto ins) { return is_reduce(*ins); });
    if(result.size() > 2)
        return {};
    // Only handle reduce_sum for now
    // TODO: Support other reduction types
    if(not std::all_of(result.begin(), result.end(), [](instruction_ref ins) {
           return ins->name() == "reduce_sum";
       }))
        return {};
    if(result.size() < 2)
        return result;
    dominator_info dom = compute_dominator(*rm);
    if(dom.strictly_dominate(result[0], result[1]))
        return {};
    if(dom.strictly_dominate(result[1], result[0]))
        return {};
    return result;
}

static std::vector<instruction_ref> get_alive(const_module_ref rm,
                                              const std::vector<instruction_ref>& splits)
{
    std::vector<instruction_ref> result;
    bool stop = false;
    liveness(*rm, [&](auto rins, const auto& live_set) {
        if(stop)
            return;
        if(rins == rm->begin())
            return;
        // We want to know what instructions are live after the split instruction
        auto ins = std::prev(rins);
        if(not contains(splits, ins))
            return;
        std::copy_if(live_set.begin(),
                     live_set.end(),
                     std::back_inserter(result),
                     [&](instruction_ref live) {
                         return live->name() != "@param" and not contains(splits, live);
                     });
        stop = true;
    });
    return result;
}

static std::string assign_op(const std::vector<instruction_ref>& splits)
{
    static std::unordered_map<std::string, std::string> m = {
        {"reduce_sum", "assign_add"},
        {"reduce_mean", "assign_add"},
        {"reduce_prod", "assign_mul"},
        {"reduce_max", "assign_max"},
        {"reduce_min", "assign_min"},
    };
    return m.at(splits.front()->name());
}

static std::vector<instruction_ref>
insert_module_inline(module& m, instruction_ref ins, const module::with_inputs& mwi)
{
    auto param_map = mwi.mod.get_ins_param_map(mwi.inputs, true);
    return m.insert_instructions(ins, &mwi.mod, &param_map);
}

static std::size_t get_reduce_size(const_module_ref rm)
{
    auto ins = std::find_if(rm->begin(), rm->end(), &is_reduce);
    assert(ins != rm->end());
    return ins->inputs().front()->get_shape().elements() / ins->get_shape().elements();
}

void split_reduce::apply(module_pass_manager& mpm) const
{
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(ins->name() != "fused_reduce")
            continue;
        auto* rm = ins->module_inputs().front();
        if(get_reduce_size(rm) < split_size)
            continue;
        auto splits = find_split(rm);
        if(splits.empty())
            continue;
        // Only use split reduce with float for now
        // TODO: Support half and other data types
        if(not std::all_of(splits.begin(), splits.end(), [](instruction_ref split) {
               return contains({shape::float_type, shape::half_type}, split->get_shape().type());
           }))
            continue;
        auto v    = ins->get_operator().to_value();
        auto axes = v["axes"].to_vector<std::int64_t>();

        auto alive = get_alive(rm, splits);

        std::array<module::with_inputs, 2> mods;
        if(not alive.empty())
        {
            auto mods3 = rm->split(ins->inputs(), alive, splits);
            auto r     = insert_module_inline(mpm.get_module(), ins, mods3[0]);
            mods3[1].replace(alive, r);
            mods3[2].replace(alive, r);
            mods = {std::move(mods3[1]), std::move(mods3[2])};
        }
        else
        {
            mods = rm->split(ins->inputs(), splits);
        }

        auto* splitm = mpm.create_module(rm->name() + "_split", std::move(mods[0].mod));
        splitm->set_bypass();

        // Insert split reduce
        auto split_reduce = mpm.get_module().insert_instruction(
            ins,
            make_op("split_fused_reduce", {{"axes", axes}, {"assign", assign_op(splits)}}),
            mods[0].inputs,
            {splitm});

        std::vector<instruction_ref> split_reduce_each;
        if(splits.size() == 1)
        {
            split_reduce_each = {split_reduce};
        }
        else
        {
            transform(range(splits.size()), std::back_inserter(split_reduce_each), [&](auto i) {
                return mpm.get_module().insert_instruction(
                    ins, make_op("get_tuple_elem", {{"index", i}}), split_reduce);
            });
        }

        mods[1].replace(splits, split_reduce_each);
        auto replaced = insert_module_inline(mpm.get_module(), ins, mods[1]);
        assert(replaced.size() == 1);
        mpm.get_module().replace_instruction(ins, replaced.front());
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
