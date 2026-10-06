/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/fuse_concat.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct fused_concat
{
    int64_t axis = 0;

    std::string name() const { return "fused_concat"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.same_ndims();
        // original concat can have multiple inputs. Let's say it has `n` input args.
        // Each of those `n` input args are converted into pointwise modules that take atleast 1
        // input parameter. Fused concat will have `n+1` module arguments. `n+1`th module is the
        // post pointwise module which can take 0 or more input arguments.
        if((inputs.size() + 1) < mods.size())
            MIGRAPHX_THROW("FUSED_CONCAT: Missing fused modules inputs parameters");
        auto input_iter = inputs.begin();
        std::vector<shape> concat_inputs;
        for(const_module_ref mod : range(mods.begin(), mods.end() - 1))
        {
            concat_inputs.push_back(*input_iter);
            input_iter += mod->get_parameter_names().size();
        }
        const_module_ref post_mod = mods.back();
        // post_mod has one input argument that is result of concat and will get generated from
        // pre-mods internally. Therefore deduct 1 from post_mod params while asserting.
        assert(input_iter + (post_mod->get_parameter_names().size() - 1) == inputs.end());
        auto type                    = std::prev(post_mod->end())->get_shape().type();
        const auto& first_shape_lens = concat_inputs.front().lens();
        auto mismatch_it =
            std::find_if_not(concat_inputs.begin() + 1, concat_inputs.end(), [&](auto s) {
                const auto& lens = s.lens();
                return std::equal(lens.begin(),
                                  lens.begin() + axis,
                                  first_shape_lens.begin(),
                                  first_shape_lens.begin() + axis) and
                       std::equal(lens.begin() + axis + 1,
                                  lens.end(),
                                  first_shape_lens.begin() + axis + 1,
                                  first_shape_lens.end());
            });
        if(mismatch_it != concat_inputs.end())
            MIGRAPHX_THROW("FUSED_CONCAT: all input dimensions should match along non-axis of " +
                           std::to_string(axis) + ": {" + to_string_range(first_shape_lens) +
                           "} != {" + to_string_range(mismatch_it->lens()) + "}");

        std::size_t new_dim_axis = transform_accumulate(
            concat_inputs.begin(), concat_inputs.end(), 0, std::plus<>{}, [&](const auto& input) {
                return input.lens()[axis];
            });
        auto new_lens  = concat_inputs.front().lens();
        new_lens[axis] = new_dim_axis;
        return shape::from_permutation(type, new_lens, find_permutation(inputs));
    }
};
MIGRAPHX_REGISTER_OP(fused_concat);

namespace {

bool is_fusable_pointwise(instruction_ref ins)
{
    return ins->name() == "pointwise" and ins->outputs().size() == 1 and
           ins->get_shape().type() != shape::tuple_type;
}

template <class... Ts>
auto fusable_pointwise(Ts... xs)
{
    return match::name("pointwise")(match::not_tuple(), xs...);
}

bool has_too_many_noops(std::size_t num_noops, std::size_t num_inputs)
{
    return num_noops > std::max(std::size_t{1}, num_inputs / 4);
}

int64_t get_concat_axis(instruction_ref ins)
{
    return ins->normalized_operator().to_value()["axis"].to<int64_t>();
}

// The inputs of the pointwise `ins` for the segment `seg` of its input `concat_ins`: the
// concat is replaced by `seg` and every other input is sliced to the segment
std::vector<instruction_ref> slice_segment_inputs(module& m,
                                                  instruction_ref ins,
                                                  instruction_ref concat_ins,
                                                  instruction_ref seg,
                                                  int64_t axis,
                                                  int64_t offset)
{
    int64_t len = seg->get_shape().lens()[axis];
    std::vector<instruction_ref> inputs;
    std::transform(
        ins->inputs().begin(),
        ins->inputs().end(),
        std::back_inserter(inputs),
        [&](instruction_ref input) {
            if(input == concat_ins)
                return seg;
            return m.insert_instruction(
                ins,
                make_op("slice",
                        {{"axes", {axis}}, {"starts", {offset}}, {"ends", {offset + len}}}),
                input);
        });
    return inputs;
}

template <std::size_t N, std::size_t Max = 2>
struct concat_counter
{
    static_assert(N < Max, "Factor N must be less than Max");
    std::shared_ptr<unsigned int> counter = std::make_shared<unsigned int>(0);
    unsigned int get_noop_counter() const
    {
        if(counter == nullptr)
            MIGRAPHX_THROW("Invalid counter");
        return N + Max * (*counter)++;
    }
};

struct find_concat_pointwise : concat_counter<0>
{
    auto matcher() const
    {
        auto pointwise_used_once = fusable_pointwise(match::used_once());
        return match::name("concat")(match::used_once(),
                                     match::any_of[match::inputs()](pointwise_used_once));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto concat_ins = r.result;

        std::vector<instruction_ref> inputs;
        size_t num_noops = 0;
        for(auto input : concat_ins->inputs())
        {
            if(input->name() == "pointwise" and input->outputs().size() == 1)
            {
                inputs.insert(inputs.end(), input->inputs().begin(), input->inputs().end());
            }
            else
            {
                num_noops++;
                inputs.push_back(input);
            }
        }
        if(has_too_many_noops(num_noops, concat_ins->inputs().size()))
        {
            return;
        }
        std::vector<module_ref> module_inputs;
        std::transform(concat_ins->inputs().begin(),
                       concat_ins->inputs().end(),
                       std::back_inserter(module_inputs),
                       [&](instruction_ref input) {
                           if(is_fusable_pointwise(input))
                           {
                               auto* pm = input->module_inputs().front();
                               return mpm.create_module("concat:" + pm->name(), *pm);
                           }
                           auto* pm = mpm.create_module("concat:noop" +
                                                        std::to_string(get_noop_counter()));
                           auto x   = pm->add_parameter("x0", shape{input->get_shape().type()});
                           pm->add_return({x});
                           return pm;
                       });
        auto* post_pm = mpm.create_module("noop:concat" + std::to_string(get_noop_counter()));
        auto x        = post_pm->add_parameter("!x0", shape{concat_ins->get_shape().type()});
        post_pm->add_return({x});
        module_inputs.push_back(post_pm);
        mpm.get_module().replace_instruction(
            concat_ins,
            make_op("fused_concat", concat_ins->normalized_operator().to_value()),
            inputs,
            module_inputs);
    }
};

struct find_pointwise_concat_pointwise : concat_counter<1>
{
    auto matcher() const
    {
        auto pointwise = fusable_pointwise(match::used_once());
        auto concat =
            match::name("concat")(match::used_once(), match::any_of[match::inputs()](pointwise));
        return fusable_pointwise(match::any_of[match::inputs()](concat.bind("concat")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins        = r.result;
        auto concat_ins = r.instructions["concat"];

        auto concat_arg = std::find(ins->inputs().begin(), ins->inputs().end(), concat_ins) -
                          ins->inputs().begin();
        std::vector<instruction_ref> inputs;
        for(auto input : concat_ins->inputs())
        {
            if(input->name() == "pointwise" and input->outputs().size() == 1)
                inputs.insert(inputs.end(), input->inputs().begin(), input->inputs().end());
            else
                inputs.push_back(input);
        }
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != concat_ins; });

        std::vector<module_ref> module_inputs;
        std::transform(concat_ins->inputs().begin(),
                       concat_ins->inputs().end(),
                       std::back_inserter(module_inputs),
                       [&](instruction_ref input) {
                           if(is_fusable_pointwise(input))
                           {
                               auto* pm = input->module_inputs().front();
                               return mpm.create_module("concat:" + pm->name(), *pm);
                           }
                           auto* pm = mpm.create_module("concat:noop" +
                                                        std::to_string(get_noop_counter()));
                           auto x  = pm->add_parameter("x0", shape{input->get_shape().type()});
                           pm->add_return({x});
                           return pm;
                       });

        auto* post_pm                  = ins->module_inputs().front();
        auto* rm                       = mpm.create_module(post_pm->name() + ":concat", *post_pm);
        std::vector<std::string> names = rm->get_parameter_names();
        std::sort(names.begin(), names.end());
        auto concat_param_name = names[concat_arg];
        auto concat_param      = rm->get_parameter(concat_param_name);
        auto param = rm->add_parameter("!" + concat_param_name, concat_param->get_shape());
        rm->replace_instruction(concat_param, param);
        rm->remove_instruction(concat_param);

        module_inputs.push_back(rm);
        mpm.get_module().replace_instruction(
            ins,
            make_op("fused_concat", concat_ins->normalized_operator().to_value()),
            inputs,
            module_inputs);
    }
};

// Split a pointwise op along the segments of a concat input whose inputs are
// all non-pointwise views (such as the two slices of a rotate-half). Each
// segment becomes a pointwise on sliced inputs, so the resulting concat of
// pointwise ops can then fuse into a single fused_concat kernel.
struct find_pointwise_concat_split
{
    auto matcher() const
    {
        auto concat = match::name("concat")(
            match::used_once(), match::none_of[match::inputs()](match::name("pointwise")));
        return fusable_pointwise(match::used_once(),
                                 match::any_of[match::inputs()](concat.bind("concat")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins        = r.result;
        auto concat_ins = r.instructions["concat"];
        if(concat_ins->inputs().size() < 2)
            return;
        if(std::count(ins->inputs().begin(), ins->inputs().end(), concat_ins) != 1)
            return;
        auto concat_op = concat_ins->normalized_operator();
        auto axis      = get_concat_axis(concat_ins);
        auto* pm       = ins->module_inputs().front();
        auto& m        = mpm.get_module();
        std::vector<instruction_ref> segments;
        int64_t offset = 0;
        for(auto seg : concat_ins->inputs())
        {
            auto inputs    = slice_segment_inputs(m, ins, concat_ins, seg, axis, offset);
            module pm_copy = *pm;
            auto* seg_pm   = mpm.create_module(
                pm->name() + ":split" + std::to_string(segments.size()), std::move(pm_copy));
            seg_pm->set_bypass();
            segments.push_back(m.insert_instruction(ins, ins->get_operator(), inputs, {seg_pm}));
            offset += seg->get_shape().lens()[axis];
        }
        m.replace_instruction(ins, concat_op, segments);
    }
};

// A pointwise on another concat is left for find_pointwise_concat_pointwise to fuse with
// that concat
bool is_fusable_concat_input(instruction_ref ins)
{
    return is_fusable_pointwise(ins) and
           std::none_of(ins->inputs().begin(), ins->inputs().end(), [](instruction_ref input) {
               return input->name() == "concat" and input->outputs().size() == 1;
           });
}

// The concat along `axis` that the pointwise `ins` can be split along
optional<instruction_ref> get_split_concat(instruction_ref ins, int64_t axis)
{
    if(not is_fusable_pointwise(ins))
        return nullopt;
    auto it = std::find_if(ins->inputs().begin(), ins->inputs().end(), [&](instruction_ref input) {
        return input->name() == "concat" and input->outputs().size() == 1 and
               get_concat_axis(input) == axis;
    });
    if(it == ins->inputs().end() or
       std::count(ins->inputs().begin(), ins->inputs().end(), *it) != 1)
        return nullopt;
    return *it;
}

// Fuse the pointwise module `post`, which uses the pointwise `seg` in `post_inputs`, into the
// module of `seg`
module::with_inputs fuse_into_segment(const module& parent,
                                      instruction_ref seg,
                                      const module& post,
                                      const std::vector<instruction_ref>& post_inputs)
{
    module pm    = *seg->module_inputs().front();
    auto map_ins = pm.get_ins_param_map(seg->inputs());
    map_ins[seg] = pm.get_returns().front();
    auto returns = pm.fuse(
        post, post_inputs, &map_ins, nullptr, [](const shape& s) { return shape{s.type()}; });
    pm.replace_return(returns);
    auto inputs = find_inputs(map_ins, &parent, &pm);
    return {std::move(pm), inputs};
}

// Fuse a concat that takes a parameter or literal along with a pointwise on another concat
// along the same axis. Each segment of the inner concat gets its own copy of the pointwise,
// fused into the segment's module when the segment is itself a pointwise, so a single
// fused_concat computes everything and reads the parameter directly instead of it being
// copied into the concat's buffer.
struct find_concat_pointwise_concat
{
    auto matcher() const
    {
        auto split_pointwise = fusable_pointwise(
            match::any_of[match::inputs()](match::name("concat")(match::used_once())));
        return match::name("concat")(
            match::used_once(),
            match::none_of[match::outputs()](match::name("pointwise")),
            match::any_of[match::inputs()](match::name("@param", "@literal")),
            match::any_of[match::inputs()](split_pointwise));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m                   = mpm.get_module();
        auto concat_ins           = r.result;
        auto axis                 = get_concat_axis(concat_ins);
        const auto& concat_inputs = concat_ins->inputs();
        auto is_split             = [&](instruction_ref input) {
            return get_split_concat(input, axis).has_value();
        };
        auto split_it = std::find_if(concat_inputs.begin(), concat_inputs.end(), is_split);
        if(split_it == concat_inputs.end())
            return;
        std::size_t num_noops =
            std::count_if(concat_inputs.begin(), concat_inputs.end(), [&](auto input) {
                return not is_split(input) and not is_fusable_concat_input(input);
            });
        if(has_too_many_noops(num_noops, concat_inputs.size()))
            return;

        // The module of the split pointwise is unique to it, so name the new modules after it
        auto prefix = (*split_it)->module_inputs().front()->name() + ":concat";
        std::vector<instruction_ref> inputs;
        std::vector<module_ref> module_inputs;
        auto add_module = [&](module mod, const std::vector<instruction_ref>& mod_inputs) {
            mod.set_bypass();
            module_inputs.push_back(
                mpm.create_module(prefix + std::to_string(module_inputs.size()), std::move(mod)));
            inputs.insert(inputs.end(), mod_inputs.begin(), mod_inputs.end());
        };
        for(auto input : concat_inputs)
        {
            if(auto split_concat = get_split_concat(input, axis))
            {
                const auto& pm = *input->module_inputs().front();
                int64_t offset = 0;
                for(auto seg : (*split_concat)->inputs())
                {
                    auto seg_inputs =
                        slice_segment_inputs(m, input, *split_concat, seg, axis, offset);
                    offset += seg->get_shape().lens()[axis];
                    if(is_fusable_concat_input(seg))
                    {
                        auto fused = fuse_into_segment(m, seg, pm, seg_inputs);
                        add_module(std::move(fused.mod), fused.inputs);
                    }
                    else
                    {
                        add_module(pm, seg_inputs);
                    }
                }
            }
            else if(is_fusable_concat_input(input))
            {
                add_module(*input->module_inputs().front(), input->inputs());
            }
            else
            {
                module noop;
                noop.add_return({noop.add_parameter("x0", shape{input->get_shape().type()})});
                add_module(std::move(noop), {input});
            }
        }
        module post;
        post.add_return({post.add_parameter("!x0", shape{concat_ins->get_shape().type()})});
        add_module(std::move(post), {});
        m.replace_instruction(concat_ins,
                              make_op("fused_concat", concat_ins->normalized_operator().to_value()),
                              inputs,
                              module_inputs);
    }
};

// Merge a same-axis concat that feeds directly into another concat
struct find_nested_concat
{
    auto matcher() const
    {
        auto concat_used_once = match::name("concat")(match::used_once());
        return match::name("concat")(match::any_of[match::inputs()](concat_used_once));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins  = r.result;
        auto axis = get_concat_axis(ins);
        std::vector<instruction_ref> args;
        fix([&](auto self, auto&& inputs) {
            for(auto&& i : inputs)
            {
                if(i->name() == "concat" and get_concat_axis(i) == axis and
                   i->outputs().size() == 1)
                    self(i->inputs());
                else
                    args.push_back(i);
            }
        })(ins->inputs());
        mpm.get_module().replace_instruction(ins, ins->normalized_operator(), args);
    }
};

} // namespace

void fuse_concat::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_pointwise_concat_split{});
    mpm.run_pass(migraphx::dead_code_elimination{});
    match::find_matches(mpm, find_nested_concat{});
    mpm.run_pass(migraphx::dead_code_elimination{});
    match::find_matches(mpm, find_concat_pointwise_concat{});
    mpm.run_pass(migraphx::dead_code_elimination{});
    match::find_matches(mpm, find_pointwise_concat_pointwise{});
    mpm.run_pass(migraphx::dead_code_elimination{});
    match::find_matches(mpm, find_concat_pointwise{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
