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
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/module.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/cpp_generator.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

static std::vector<std::size_t> vector_sizes(const std::vector<shape>& inputs)
{
    // If all inputs are half then only use half2
    if(std::all_of(inputs.begin(), inputs.end(), [](const auto& s) {
           return s.type() == shape::half_type;
       }))
        return {2};
    return {4, 2};
}

vectorize vectorize::elements(std::size_t axis,
                              const std::vector<shape>& inputs,
                              const std::vector<std::size_t>& sizes)
{
    if(std::all_of(
           inputs.begin(), inputs.end(), [&](const auto& s) { return s.lens()[axis] == 1; }))
        return {1, axis};
    std::vector<std::size_t> max_vec_size;
    std::transform(inputs.begin(),
                   inputs.end(),
                   std::back_inserter(max_vec_size),
                   [&](const auto& input) -> std::size_t {
                       auto stride = input.strides()[axis];
                       auto len    = input.lens()[axis];
                       if(not contains({0, 1}, stride))
                           return 1;
                       if(len == 1 and input.elements() > sizes.front())
                           return sizes.front();
                       auto it = std::find_if(sizes.begin(), sizes.end(), [&](auto vsize) {
                           // The len is divisible by the size and all the strides are divisible by
                           // the size
                           return (len % vsize) == 0 and
                                  std::all_of(
                                      input.strides().begin(), input.strides().end(), [&](auto i) {
                                          return contains({0, 1}, i) or i % vsize == 0;
                                      });
                       });
                       if(it != sizes.end())
                           return *it;
                       return 1;
                   });
    return {*std::min_element(max_vec_size.begin(), max_vec_size.end()), axis};
}

vectorize vectorize::elements(context& ctx, std::size_t axis, const std::vector<shape>& inputs)
{
    if(inputs.empty())
        return {1, axis};
    std::size_t n = std::max_element(inputs.begin(),
                                     inputs.end(),
                                     by(std::less<>{}, [](const auto& s) { return s.elements(); }))
                        ->elements();
    std::size_t max_global = ctx.get_current_device().get_cu_count() *
                             ctx.get_current_device().get_max_workitems_per_cu();
    std::size_t over = n / max_global;
    bool broadcasted =
        std::any_of(inputs.begin(), inputs.end(), [](const auto& s) { return s.broadcasted(); });
    std::vector<std::size_t> sizes;
    if(broadcasted and over > 8)
        sizes.push_back(8);
    if(over > 4)
        sizes.push_back(4);
    sizes.push_back(2);
    return elements(axis, inputs, sizes);
}

vectorize vectorize::elements(std::size_t axis, const std::vector<shape>& inputs)
{
    return elements(axis, inputs, vector_sizes(inputs));
}

std::string vectorize::str() const
{
    return "vectorize<" + to_string(size) + ", " + to_string(axis) + ">()";
}

preload preload::broadcasts(std::size_t axis, const std::vector<shape>& inputs)
{
    const std::size_t max_lds_bytes = 4096;
    std::vector<bool> result(inputs.size());
    std::vector<std::size_t> preloaded;
    auto idxs = range(inputs.size());
    std::copy_if(idxs.begin(), idxs.end(), std::back_inserter(preloaded), [&](auto i) {
        return inputs[i].strides()[axis] == 0;
    });
    std::sort(preloaded.begin(), preloaded.end(), by(std::less<>{}, [&](auto i) {
                  return inputs[i].bytes();
              }));

    std::size_t bytes = 0;
    for(auto i : preloaded)
    {
        const auto& input = inputs[i];
        bytes += input.bytes();
        if(bytes > max_lds_bytes)
            break;
        result[i] = true;
    }
    return {result};
}

std::string preload::str() const
{
    std::vector<std::string> bool_strs;
    std::transform(args.begin(), std::prev(args.end()), std::back_inserter(bool_strs), [](bool b) {
        if(b)
            return "true";
        return "false";
    });
    return "auto_preload<false, " + join_strings(bool_strs, ", ") + ">(idx)";
}

bool preload::is_preloading() const
{
    return std::accumulate(args.begin(), args.end(), false, std::logical_or<>{});
}

std::size_t find_fast_axis(const std::vector<shape>& inputs)
{
    auto permutation = find_permutation(inputs);
    auto it          = std::max_element(permutation.begin(), permutation.end());
    return it - permutation.begin();
}

std::string make_transformer_args(std::vector<std::string> transformers)
{
    return join_strings(std::move(transformers), ", ");
}

void generate_pointwise(cpp_generator& gg, const module& pm, const std::string& name)
{
    module m = pm;
    run_passes(m,
               {rewrite_quantization{}, eliminate_common_subexpression{}, dead_code_elimination{}});
    cpp_generator g;
    g.fmap([](const std::string& fname) { return "migraphx::" + fname; });
    g.add_point_op("where", "${function:where}(${0}, ${1}, ${2})");
    g.add_point_op("prelu", "${function:where}(${0} < 0, ${0} * ${1}, ${0})");
    g.add_point_op("sign", "${function:where}(${0} > 0, 1, ${function:where}(${0} < 0, -1, 0))");
    g.add_point_op("equal", "migraphx::abs(${0} == ${1})");
    g.add_point_op("less", "migraphx::abs(${0} < ${1})");
    g.add_point_op("greater", "migraphx::abs(${0} > ${1})");
    g.add_point_op("not", "migraphx::abs(not ${0})");
    // Add explict conversions
    g.fresult(
        [](const shape& s) { return "migraphx::convert<" + shape::cpp_type(s.type()) + ">"; });
    gg.create_function(g.generate_module(m)
                           .set_attributes({"__device__", "__attribute__((const))"})
                           .set_generic_types(m)
                           .set_name(name));
}
std::string generate_pointwise(const module& pm, const std::string& name)
{
    cpp_generator g;
    generate_pointwise(g, pm, name);
    return g.str();
}

std::string reduce_op::str() const
{
    return write + "(r.reduce(" + reduction + ", " + init + ", " + read + ")(" + input + "))";
}
void reduce_op::set(instruction_ref ins, const operation& op)
{
    if(op.name() == "reduce_sum")
    {
        reduction = "op::sum{}";
    }
    else if(op.name() == "reduce_mean")
    {
        auto s               = ins->inputs().front()->get_shape();
        auto reduce_elements = s.elements() / ins->get_shape().elements();
        auto reduce_type     = s.type();
        reduction            = "op::sum{}";
        std::string mean     = "op::mean<" + std::to_string(reduce_elements) + ">{}";
        // Use float accumulator when reduction size is too large for half
        if(reduce_type == shape::half_type and reduce_elements > 16384)
            read = "compose(" + mean + ", op::convert_to<float>{})";
        else if(contains({shape::float_type, shape::half_type, shape::double_type}, reduce_type))
            read = mean;
        else
            write = mean;
    }
    else if(op.name() == "reduce_max")
    {
        reduction = "op::max{}";
        init      = "lowest{}";
    }
    else if(op.name() == "reduce_min")
    {
        reduction = "op::min{}";
        init      = "highest{}";
    }
    else if(op.name() == "reduce_prod")
    {
        reduction = "op::product{}";
        init      = "1";
    }
    else
    {
        MIGRAPHX_THROW("Unsupported reduce");
    }
}
std::string reduce_op::generate(instruction_ref ins, const std::string& x)
{
    reduce_op r{x};
    r.set(ins, ins->get_operator());
    return r.str();
}

static bool use_lazy_inner(instruction_ref ins)
{
    if(ins->outputs().size() != 1)
        return false;
    auto output = ins->outputs().front();
    return contains(output->name(), "reduce") or output->name() == "@return";
}

std::string generate_reduce(const module& m, const std::string& name)
{
    cpp_generator g;
    auto ilens    = m.get_parameter_shapes().begin()->second.lens();
    std::size_t i = 0;
    auto f        = g.generate_module(m, [&](instruction_ref ins, const auto& names) {
        if(contains(ins->name(), "reduce"))
        {
            return reduce_op::generate(ins, names.at(ins->inputs().front()));
        }
        else if(ins->name() == "pointwise")
        {
            auto pointwise_name = "pointwise" + std::to_string(i);
            i++;
            generate_pointwise(g, *ins->module_inputs().front(), pointwise_name);
            std::vector<instruction_ref> tensors;
            std::copy_if(ins->inputs().begin(),
                         ins->inputs().end(),
                         std::back_inserter(tensors),
                         [&](auto input) {
                             return input->get_shape().lens() == ilens and
                                    not input->get_shape().broadcasted();
                         });
            auto inner_names = names;
            for(auto input : ins->inputs())
            {
                if(input->name() != "@param")
                    continue;
                if(contains(tensors, input))
                    continue;
                inner_names[input] += "[out_idx]";
            }
            for(auto input : tensors)
                inner_names[input] += "_lambda_param";
            auto call_function =
                pointwise_name + "(" +
                join_strings(cpp_generator::to_args(ins->inputs(), inner_names), ", ") + ")";
            if(tensors.empty())
                return call_function;
            const std::string inner_template =
                "r.${inner}([=](${params}) { return ${call}; })(${args})";
            std::string inner_name = use_lazy_inner(ins) ? "lazy_inner" : "inner";
            auto args              = cpp_generator::to_args(tensors, names);
            auto params            = cpp_generator::to_args(tensors, inner_names);
            std::transform(
                params.begin(), params.end(), params.begin(), [](auto s) { return "auto " + s; });
            return interpolate_string(inner_template,
                                      {{"inner", inner_name},
                                       {"params", join_strings(params, ", ")},
                                       {"args", join_strings(args, ", ")},
                                       {"call", call_function}});
        }
        else if(ins->name() == "multibroadcast")
        {
            return names.at(ins->inputs().front());
        }
        MIGRAPHX_THROW("Unknown operator: " + ins->name());
    });
    f.set_attributes({"__device__", "__attribute__((const))"}).set_generic_types(m).set_name(name);
    f.add_generic_param("r");
    f.add_generic_param("out_idx");
    f.unused_param("out_idx");
    g.create_function(f);
    return g.str();
}

static std::vector<std::string> get_op_names(const module& m)
{
    std::vector<std::string> result;
    for(auto& ins : m)
    {
        if(starts_with(ins.name(), "@"))
            continue;
        if(contains({"multibroadcast", "contiguous", "identity"}, ins.name()))
            continue;
        if(ins.name() == "pointwise")
        {
            auto names = get_op_names(*ins.module_inputs().front());
            result.insert(result.end(), names.begin(), names.end());
        }
        else
        {
            result.push_back(ins.name());
        }
    }
    return result;
}

std::string generate_name_from_ops(const module& m, const std::string& postname)
{
    auto op_names = get_op_names(m);
    if(not postname.empty())
        op_names.push_back(postname);
    return join_strings(op_names, "_");
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
