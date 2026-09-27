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
 *
 */
#include <migraphx/gpu/prepare_reduce.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/module.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/tune_axis.hpp>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct parallel_reduce
{
    operation op = op::identity{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::parallel_reduce"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        std::vector<shape> result;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(result), [&](auto input) {
            return op.compute_shape({input});
        });
        return shape{result};
    }
};
MIGRAPHX_REGISTER_OP(parallel_reduce);

struct arg_reduce
{
    operation op = op::identity{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::arg_reduce"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        auto index_shape = op.compute_shape({inputs.front()});
        auto value_shape = index_shape.with_type(inputs.front().type());
        return shape{{value_shape, index_shape}};
    }
};
MIGRAPHX_REGISTER_OP(arg_reduce);

struct make_indices
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::make_indices"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        if(inputs.size() != 1)
            MIGRAPHX_THROW("gpu::make_indices expects one value tensor operand");
        return shape{shape::uint32_type, inputs.front().lens()};
    }
};
MIGRAPHX_REGISTER_OP(make_indices);

// unpack_int4 followed by a convert to a float type, with an integer bias
// (the negated zero point) folded in: value = nibble + bias
struct unpack_int4_convert
{
    int64_t axis              = -1;
    shape::type_t target_type = shape::half_type;
    double bias               = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"), f(self.target_type, "target_type"), f(self.bias, "bias"));
    }

    std::string name() const { return "gpu::unpack_int4_convert"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        return make_op("unpack_int4", {{"axis", axis}})
            .compute_shape(inputs)
            .with_type(target_type);
    }
};
MIGRAPHX_REGISTER_OP(unpack_int4_convert);

namespace {

// find argmin/argmax operations
std::vector<instruction_ref> find_arg_reduce(module& m)
{
    std::vector<instruction_ref> result;
    auto im = iterator_for(m);
    std::copy_if(im.begin(), im.end(), std::back_inserter(result), [](auto ins) {
        return ins->name() == "argmin" or ins->name() == "argmax";
    });
    return result;
}

// rewrite argmin/argmax to return lazy indices and values tuple
void rewrite_arg_reduce(module& m)
{
    for(auto ins : find_arg_reduce(m))
    {
        auto input     = ins->inputs().front();
        // make_indices(value): lazy index stream sized from the same tensor the reducer slices
        auto indices = m.insert_instruction(ins, make_indices{}, input);
        // arg_reduce op to get values and indices tuple
        auto arg_reduce_ins =
            m.insert_instruction(ins, arg_reduce{ins->get_operator()}, input, indices);
        auto result =
            m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", 1}}), arg_reduce_ins);
        m.replace_instruction(ins, result);
    }
}

std::vector<instruction_ref> find_reduce(module& m)
{
    std::vector<instruction_ref> result;
    auto im = iterator_for(m);
    std::copy_if(im.begin(), im.end(), std::back_inserter(result), [](auto ins) {
        if(contains({"gpu::parallel_reduce", "reduce_mean", "gpu::arg_reduce"}, ins->name()))
            return false;
        return contains(ins->name(), "reduce");
    });
    return result;
}

std::vector<instruction_ref> find_parallel_reduce(const std::vector<instruction_ref>& r)
{
    std::vector<instruction_ref> result;
    auto ir = iterator_for(r);
    transform_if(
        ir.begin(),
        ir.end(),
        std::back_inserter(result),
        [&](auto x) {
            return std::none_of(
                std::next(x), r.end(), [&](auto reduce) { return reaches(*x, reduce); });
        },
        [](auto x) { return *x; });
    return result;
}

void fuse_reductions(module& m)
{
    auto rs = find_parallel_reduce(find_reduce(m));
    if(rs.size() < 2)
        return;
    // Only handle the same reduction operator (and its data-type) for now
    if(std::any_of(std::next(rs.cbegin()), rs.cend(), [&](auto r) {
           return (*rs.cbegin())->name() != r->name() or
                  (*rs.cbegin())->get_shape().type() != r->get_shape().type();
       }))
        return;

    auto last = rs.front();
    auto op   = last->get_operator();
    std::vector<instruction_ref> inputs;
    std::transform(rs.begin(), rs.end(), std::back_inserter(inputs), [&](auto r) {
        return r->inputs().front();
    });
    auto pr = m.insert_instruction(last, parallel_reduce{op}, inputs);
    int i   = 0;
    for(auto r : rs)
    {
        m.replace_instruction(r, make_op("get_tuple_elem", {{"index", i}}), pr);
        i++;
    }
    m.sort();
}

/// The parameters of the pointwise module that read `input`
std::vector<instruction_ref>
find_pointwise_params(const module& pm, instruction_ref ins, instruction_ref input)
{
    auto pmap = pm.get_ins_param_map(ins->inputs(), true);
    std::vector<instruction_ref> result;
    transform_if(
        pmap.begin(),
        pmap.end(),
        std::back_inserter(result),
        [&](const auto& p) { return p.second == input; },
        [](const auto& p) { return p.first; });
    return result;
}

/// The converts reading the pointwise parameters fed by `unpack`, or nothing
/// when a consumer is not a pointwise or uses the values unconverted
optional<std::vector<instruction_ref>> find_unpack_converts(instruction_ref unpack)
{
    std::vector<instruction_ref> converts;
    for(auto out : unpack->outputs())
    {
        auto input                             = unpack;
        std::vector<instruction_ref> consumers = {out};
        if(out->name() == "multibroadcast")
        {
            input     = out;
            consumers = out->outputs();
        }
        for(auto consumer : consumers)
        {
            if(consumer->name() != "pointwise")
                return nullopt;
            const auto& pm = *consumer->module_inputs().front();
            for(auto param : find_pointwise_params(pm, consumer, input))
            {
                for(auto use : param->outputs())
                {
                    if(use->name() != "convert")
                        return nullopt;
                    converts.push_back(use);
                }
            }
        }
    }
    return converts;
}

/// The integer literal added to `ins` by its only consumer (an add, or a sub
/// with `ins` first); only a small integer bias folds into the unpack exactly
optional<double> find_literal_bias(instruction_ref ins)
{
    if(ins->outputs().size() != 1)
        return nullopt;
    auto op     = ins->outputs().front();
    bool is_add = op->name() == "add";
    bool is_sub = op->name() == "sub" and op->inputs()[0] == ins;
    if((not is_add and not is_sub) or op->inputs().size() != 2)
        return nullopt;
    auto lit = op->inputs()[0] == ins ? op->inputs()[1] : op->inputs()[0];
    if(lit->name() != "@literal" or lit->get_shape().elements() != 1)
        return nullopt;
    auto value = lit->get_literal().at<double>();
    if(is_sub)
        value = -value;
    if(not float_equal(value, std::floor(value)) or std::fabs(value) > 256)
        return nullopt;
    return value;
}

struct unpack_convert
{
    shape::type_t type = shape::half_type;
    // Set when the zero point add of every consumer folds into the unpack
    optional<double> bias = nullopt;
};

/// Whether `unpack` can fold its converts: every consumer must convert to the
/// same fp16 or fp32 type; the zero point folds too when all add the same literal
optional<unpack_convert> fold_decision(instruction_ref unpack)
{
    auto converts = find_unpack_converts(unpack);
    if(not converts.has_value() or converts->empty())
        return nullopt;
    unpack_convert result;
    result.type = converts->front()->get_shape().type();
    if(not contains({shape::half_type, shape::float_type}, result.type))
        return nullopt;
    if(std::any_of(converts->begin(), converts->end(), [&](auto convert) {
           return convert->get_shape().type() != result.type;
       }))
        return nullopt;
    auto bias = find_literal_bias(converts->front());
    if(not bias.has_value())
        return result;
    if(std::all_of(converts->begin(), converts->end(), [&](auto convert) {
           auto b = find_literal_bias(convert);
           return b.has_value() and float_equal(*b, *bias);
       }))
        result.bias = bias;
    return result;
}

/// The pointwise module with the folded parameters typed as their converted
/// values, read in place of the convert and of the folded zero point add
module fold_pointwise_module(const module& pm,
                             const std::unordered_map<instruction_ref, unpack_convert>& folded)
{
    module result;
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    // Re-add the parameters first, in their original order, so the pointwise
    // inputs still map to them by position
    for(const auto& name : pm.get_parameter_names())
    {
        auto ins = pm.get_parameter(name);
        auto it  = folded.find(ins);
        auto s   = ins->get_shape();
        if(it != folded.end())
            s = s.with_type(it->second.type);
        auto param   = result.add_parameter(name, s);
        map_ins[ins] = param;
        if(it == folded.end())
            continue;
        for(auto convert : ins->outputs())
        {
            map_ins[convert] = param;
            if(it->second.bias.has_value())
                map_ins[convert->outputs().front()] = param;
        }
    }
    result.add_return(result.add_instructions(&pm, &map_ins));
    return result;
}

/// The gpu::unpack_int4_convert standing in for `unpack`, created once
instruction_ref convert_unpack(module& m,
                               std::unordered_map<instruction_ref, instruction_ref>& converted,
                               instruction_ref unpack,
                               const unpack_convert& uc)
{
    auto it = converted.find(unpack);
    if(it != converted.end())
        return it->second;
    auto axis = tune_axis(unpack->get_shape().ndim(),
                          unpack->get_operator().to_value().at("axis").to<int>(),
                          unpack->name());
    auto ins  = m.insert_instruction(
        unpack, unpack_int4_convert{axis, uc.type, uc.bias.value_or(0)}, unpack->inputs());
    converted[unpack] = ins;
    return ins;
}

/// The unpack a pointwise input reads, looking through a broadcast
instruction_ref unpack_source(instruction_ref input)
{
    if(input->name() == "multibroadcast")
        return input->inputs().front();
    return input;
}

// Replace the unpacks whose consumers convert them with gpu::unpack_int4_convert
// and rebuild those pointwise modules to read the converted values. Expects the
// pointwise modules to have quantization rewritten and simplified, so the zero
// point is a literal add after the convert
void fold_unpack_convert(module_pass_manager& mpm)
{
    auto& m = mpm.get_module();
    std::unordered_map<instruction_ref, unpack_convert> unpacks;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "unpack_int4")
            continue;
        auto uc = fold_decision(ins);
        if(uc.has_value())
            unpacks[ins] = *uc;
    }
    if(unpacks.empty())
        return;
    std::unordered_map<instruction_ref, instruction_ref> converted;
    // A module shared by several pointwise instructions is rebuilt for the first only
    std::unordered_set<module_ref> rebuilt;
    for(auto pw : iterator_for(m))
    {
        if(pw->name() != "pointwise")
            continue;
        auto* pm = pw->module_inputs().front();
        if(contains(rebuilt, pm))
            continue;
        std::unordered_map<instruction_ref, unpack_convert> folded;
        for(const auto& [param, input] : pm->get_ins_param_map(pw->inputs(), true))
        {
            auto it = unpacks.find(unpack_source(input));
            if(it != unpacks.end())
                folded[param] = it->second;
        }
        if(folded.empty())
            continue;
        rebuilt.insert(pm);
        auto inputs = pw->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            auto it = unpacks.find(unpack_source(input));
            if(it == unpacks.end())
                return input;
            auto unpack = convert_unpack(m, converted, it->first, it->second);
            if(input->name() == "multibroadcast")
                unpack = m.insert_instruction(input, input->get_operator(), unpack);
            return unpack;
        });
        auto* fm = mpm.create_module(pm->name() + ":unpack", fold_pointwise_module(*pm, folded));
        m.replace_instruction(pw, pw->get_operator(), inputs, {fm});
    }
}

} // namespace

void prepare_reduce::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();
    // rewrite argmin/argmax to handle tuples
    rewrite_arg_reduce(m);
    fuse_reductions(m);
    fold_unpack_convert(mpm);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
