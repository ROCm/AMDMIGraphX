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
#include <migraphx/instruction.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
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
        // original concat can have multiple inputs. Let's say it has `n` input args.
        // Each of those `n` input args are converted into pointwise modules that take atleast 1
        // input parameter. Fused concat will have `n+1` module arguments. `n+1`th module is the
        // post pointwise module which can take 0 or more input arguments.
        if(mods.size() < 2)
            MIGRAPHX_THROW("FUSED_CONCAT: Missing fused modules");
        auto expected_inputs = transform_accumulate(
            mods.begin(), mods.end() - 1, std::size_t{0}, std::plus<>{}, [](const_module_ref mod) {
                auto nparams = mod->get_parameter_names().size();
                if(nparams == 0)
                    MIGRAPHX_THROW("FUSED_CONCAT: Pre-concat module requires an input parameter");
                return nparams;
            });
        const_module_ref post_mod = mods.back();
        auto post_params          = post_mod->get_parameter_names().size();
        if(post_params == 0)
            MIGRAPHX_THROW("FUSED_CONCAT: Post-concat module requires a concat parameter");
        expected_inputs += post_params - 1;
        if(expected_inputs != inputs.size())
            MIGRAPHX_THROW("FUSED_CONCAT: Expected " + std::to_string(expected_inputs) +
                           " inputs for fused modules but got " + std::to_string(inputs.size()));

        check_shapes{inputs, *this}.same_ndims();
        auto input_iter = inputs.begin();
        std::vector<shape> concat_inputs;
        for(const_module_ref mod : range(mods.begin(), mods.end() - 1))
        {
            concat_inputs.push_back(*input_iter);
            input_iter += mod->get_parameter_names().size();
        }
        // post_mod has one input argument that is result of concat and will get generated from
        // pre-mods internally. Therefore deduct 1 from post_mod params.
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
        if(num_noops > std::max(size_t{1}, concat_ins->inputs().size() / 4))
        {
            return;
        }
        std::vector<module_ref> module_inputs;
        auto module_prefix = mpm.get_module().name() + ":";
        std::transform(concat_ins->inputs().begin(),
                       concat_ins->inputs().end(),
                       std::back_inserter(module_inputs),
                       [&](instruction_ref input) {
                           if(is_fusable_pointwise(input))
                           {
                               auto* pm = input->module_inputs().front();
                               return mpm.create_module("concat:" + pm->name(), *pm);
                           }
                           auto* pm = mpm.create_module(module_prefix + "concat:noop" +
                                                        std::to_string(get_noop_counter()));
                           auto x   = pm->add_parameter("x0", shape{input->get_shape().type()});
                           pm->add_return({x});
                           return pm;
                       });
        auto* post_pm =
            mpm.create_module(module_prefix + "noop:concat" + std::to_string(get_noop_counter()));
        auto x = post_pm->add_parameter("!x0", shape{concat_ins->get_shape().type()});
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
        auto module_prefix = mpm.get_module().name() + ":";
        std::transform(concat_ins->inputs().begin(),
                       concat_ins->inputs().end(),
                       std::back_inserter(module_inputs),
                       [&](instruction_ref input) {
                           if(is_fusable_pointwise(input))
                           {
                               auto* pm = input->module_inputs().front();
                               return mpm.create_module("concat:" + pm->name(), *pm);
                           }
                           auto* pm = mpm.create_module(module_prefix + "concat:noop" +
                                                        std::to_string(get_noop_counter()));
                           auto x   = pm->add_parameter("x0", shape{input->get_shape().type()});
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
        auto axis      = concat_op.to_value()["axis"].to<int64_t>();
        auto* pm       = ins->module_inputs().front();
        auto& m        = mpm.get_module();
        std::vector<instruction_ref> segments;
        int64_t offset = 0;
        for(auto seg : concat_ins->inputs())
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
            module pm_copy = *pm;
            auto* seg_pm   = mpm.create_module(
                pm->name() + ":split" + std::to_string(segments.size()), std::move(pm_copy));
            seg_pm->set_bypass();
            segments.push_back(m.insert_instruction(ins, ins->get_operator(), inputs, {seg_pm}));
            offset += len;
        }
        m.replace_instruction(ins, concat_op, segments);
    }
};

// One GPU concat kernel is compiled as a parameter pack. A few hundred inputs
// overflows that compiler, which is what the unrolled segmentation concats hit.
constexpr std::size_t max_concat_inputs = 32;

int64_t concat_axis(instruction_ref ins)
{
    return ins->normalized_operator().to_value()["axis"].to<int64_t>();
}

// Inline same-axis, single-use concat inputs. Iterative so a chain of hundreds
// of concats does not recurse the call stack.
std::vector<instruction_ref> flatten_concat(instruction_ref ins, int64_t axis)
{
    std::vector<instruction_ref> args;
    std::vector<instruction_ref> pending;
    auto enqueue = [&](instruction_ref concat) {
        auto inputs = concat->inputs();
        pending.insert(pending.end(), inputs.rbegin(), inputs.rend());
    };
    enqueue(ins);
    while(not pending.empty())
    {
        auto input = pending.back();
        pending.pop_back();
        if(input->name() == "concat" and concat_axis(input) == axis and
           input->outputs().size() == 1)
            enqueue(input);
        else
            args.push_back(input);
    }
    return args;
}

// Rebuild as a tree of concats that each stay within the kernel input limit.
void replace_with_chunked_concat(module& m,
                                 instruction_ref ins,
                                 std::vector<instruction_ref> args)
{
    auto op = ins->normalized_operator();
    while(args.size() > max_concat_inputs)
    {
        std::vector<instruction_ref> next;
        next.reserve((args.size() + max_concat_inputs - 1) / max_concat_inputs);
        for(std::size_t i = 0; i < args.size();)
        {
            auto n = std::min(max_concat_inputs, args.size() - i);
            if(n == 1)
                next.push_back(args[i]);
            else
            {
                std::vector<instruction_ref> chunk(args.begin() + i, args.begin() + i + n);
                next.push_back(m.insert_instruction(ins, op, chunk));
            }
            i += n;
        }
        args = std::move(next);
    }
    m.replace_instruction(ins, op, args);
}

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
        auto axis = concat_axis(ins);
        replace_with_chunked_concat(mpm.get_module(), ins, flatten_concat(ins, axis));
    }
};

// A single concat can already be wider than the kernel limit.
struct find_wide_concat
{
    auto matcher() const { return match::name("concat"); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        if(ins->inputs().size() <= max_concat_inputs)
            return;
        std::vector<instruction_ref> args(ins->inputs().begin(), ins->inputs().end());
        replace_with_chunked_concat(mpm.get_module(), ins, std::move(args));
    }
};

} // namespace

void fuse_concat::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_pointwise_concat_split{});
    mpm.run_pass(migraphx::dead_code_elimination{});
    match::find_matches(mpm, find_nested_concat{});
    mpm.run_pass(migraphx::dead_code_elimination{});
    match::find_matches(mpm, find_wide_concat{});
    mpm.run_pass(migraphx::dead_code_elimination{});
    match::find_matches(mpm, find_pointwise_concat_pointwise{});
    mpm.run_pass(migraphx::dead_code_elimination{});
    match::find_matches(mpm, find_concat_pointwise{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
