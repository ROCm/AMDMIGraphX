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
#ifndef MIGRAPHX_GUARD_OPERATORS_SELECT_MODULE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SELECT_MODULE_HPP

#include <migraphx/algorithm.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/context.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/zip_view.hpp>
#include <numeric>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct select_module
{
    shape output_dyn_shapes;
    optional<shape> logical_output_dyn_shapes;
    std::size_t num_inputs = 0;
    std::vector<shape> input_shapes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_dyn_shapes, "output_dyn_shapes"),
                    f(self.logical_output_dyn_shapes, "logical_output_dyn_shapes"),
                    f(self.num_inputs, "num_inputs"),
                    f(self.input_shapes, "input_shapes"));
    }

    std::string name() const { return "select_module"; }

    std::size_t num_outputs() const { return output_dyn_shapes.sub_shapes().size(); }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>&) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);
        if(logical_output_dyn_shapes.has_value() and
           logical_output_dyn_shapes->sub_shapes().size() != num_outputs())
            MIGRAPHX_THROW("SELECT_MODULE: logical output count does not match outputs");
        if(logical_output_dyn_shapes.has_value())
        {
            if(num_inputs > inputs.size())
                MIGRAPHX_THROW("SELECT_MODULE: input count exceeds instruction inputs");
            std::unordered_set<sym::expr> input_variables;
            std::for_each(inputs.begin(), inputs.begin() + num_inputs, [&](const shape& input) {
                if(not input.symbolic())
                    return;
                transform_if(
                    input.dyn_dims().begin(),
                    input.dyn_dims().end(),
                    std::inserter(input_variables, input_variables.end()),
                    [](const auto& dim) { return dim.sym_expr.name() == "variable"; },
                    [](const auto& dim) { return sym::as_symbol(dim.sym_expr); });
            });
            migraphx::for_each(
                logical_output_dyn_shapes->sub_shapes().begin(),
                logical_output_dyn_shapes->sub_shapes().end(),
                output_dyn_shapes.sub_shapes().begin(),
                [&](const shape& logical, const shape& output) {
                    if(logical.dynamic() and not logical.symbolic())
                        MIGRAPHX_THROW("SELECT_MODULE: logical outputs must be static or symbolic");
                    auto logical_max = logical.max_lens();
                    auto output_max  = output.max_lens();
                    if(logical.type() != output.type() or logical.ndim() != output.ndim() or
                       not std::equal(logical_max.begin(),
                                      logical_max.end(),
                                      output_max.begin(),
                                      [](auto logical_len, auto output_len) {
                                          return logical_len <= output_len;
                                      }))
                        MIGRAPHX_THROW(
                            "SELECT_MODULE: logical output is incompatible with declared output");
                    if(logical.symbolic())
                    {
                        auto has_missing_source = [&](const sym::expr& expression) {
                            auto variables = sym::find_variables(expression);
                            return any_of(variables, [&](const auto& variable) {
                                return not contains(input_variables, variable);
                            });
                        };
                        if(any_of(
                               logical.dyn_dims(),
                               [&](const auto& dim) { return has_missing_source(dim.sym_expr); }) or
                           any_of(logical.dyn_strides(), has_missing_source))
                            MIGRAPHX_THROW(
                                "SELECT_MODULE: logical output symbol has no input source");
                    }
                });
        }
        return shape{output_dyn_shapes};
    }

    void finalize(context&, const shape&, const std::vector<shape>& inputs)
    {
        if(num_inputs > inputs.size())
            MIGRAPHX_THROW("SELECT_MODULE: input count exceeds instruction inputs");
        input_shapes.assign(inputs.begin(), inputs.begin() + num_inputs);
    }

    std::vector<shape> compute_logical_output_shapes(const std::vector<argument>& args) const
    {
        if(not logical_output_dyn_shapes.has_value())
            return {};
        if(input_shapes.size() != num_inputs or args.size() < num_inputs)
            MIGRAPHX_THROW("SELECT_MODULE: symbolic input shapes were not finalized");

        std::unordered_map<sym::expr, std::size_t> values;
        migraphx::for_each(
            input_shapes.begin(),
            input_shapes.end(),
            args.begin(),
            [&](const auto& expected, const auto& arg) {
                const auto& actual = arg.get_shape();
                if(expected.dynamic())
                {
                    if(actual.type() != expected.type() or
                       not shape::is_compatible_lens(actual, expected))
                        MIGRAPHX_THROW(
                            "SELECT_MODULE: runtime input does not match symbolic input shape");
                }
                else if(actual != expected)
                {
                    MIGRAPHX_THROW(
                        "SELECT_MODULE: runtime input does not match static input shape");
                }
                if(not expected.symbolic())
                    return;
                if(expected.ndim() != actual.ndim())
                    MIGRAPHX_THROW(
                        "SELECT_MODULE: runtime input rank does not match symbolic input");
                migraphx::for_each(expected.dyn_dims().begin(),
                                   expected.dyn_dims().end(),
                                   actual.lens().begin(),
                                   [&](const auto& dim, auto len) {
                                       if(dim.sym_expr.name() != "variable")
                                           return;
                                       auto variable = sym::as_symbol(dim.sym_expr);
                                       auto result   = values.emplace(variable, len);
                                       if(not result.second and result.first->second != len)
                                           MIGRAPHX_THROW("SELECT_MODULE: repeated symbol has "
                                                          "inconsistent runtime dimensions");
                                   });
            });

        std::vector<shape> result;
        std::transform(logical_output_dyn_shapes->sub_shapes().begin(),
                       logical_output_dyn_shapes->sub_shapes().end(),
                       std::back_inserter(result),
                       [&](const shape& s) { return s.symbolic() ? s.to_static(values) : s; });
        return result;
    }

    std::vector<std::string> get_input_parameter_names(module_ref mod) const
    {
        auto param_names = mod->get_parameter_names();
        std::vector<std::string> ret;
        std::copy_if(param_names.cbegin(),
                     param_names.cend(),
                     std::back_inserter(ret),
                     [](const auto& pn) { return not contains(pn, "#output_"); });
        std::sort(ret.begin(), ret.end());
        return ret;
    }

    std::vector<std::string> get_output_parameter_names(module_ref mod) const
    {
        auto param_names = mod->get_parameter_names();
        std::vector<std::string> ret;
        std::copy_if(param_names.cbegin(),
                     param_names.cend(),
                     std::back_inserter(ret),
                     [](const auto& pn) { return contains(pn, "#output_"); });
        // needs to be sorted to ensure output parameter ordering
        std::sort(ret.begin(), ret.end());
        return ret;
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        auto logical_outputs = compute_logical_output_shapes(args);
        if(not logical_outputs.empty() and
           std::all_of(logical_outputs.begin(), logical_outputs.end(), [](const shape& s) {
               return s.elements() == 0;
           }))
        {
            std::vector<argument> results;
            bool has_output_allocations = args.size() == num_inputs + num_outputs();
            auto output_start           = has_output_allocations ? args.size() - num_outputs() : 0;
            auto output_indices         = range(logical_outputs.size());
            std::transform(output_indices.begin(),
                           output_indices.end(),
                           std::back_inserter(results),
                           [&](auto i) {
                               const auto& s = logical_outputs.at(i);
                               return has_output_allocations ? args.at(output_start + i).reshape(s)
                                                             : argument{s, nullptr};
                           });
            return argument{results};
        }

        // Input arguments are ordered like the sorted input parameters.
        auto module_iter =
            std::find_if(submodule_list.cbegin(), submodule_list.cend(), [&](module_ref mr) {
                auto in_param_names = get_input_parameter_names(mr);
                auto param_shapes   = mr->get_parameter_shapes();
                assert(in_param_names.size() <= args.size());
                return std::equal(in_param_names.cbegin(),
                                  in_param_names.cend(),
                                  args.cbegin(),
                                  [&](const auto& p_name, const auto& a) {
                                      const auto& actual   = a.get_shape();
                                      const auto& expected = param_shapes.at(p_name);
                                      if(expected.dynamic())
                                          return actual.type() == expected.type() and
                                                 shape::is_compatible_lens(actual, expected);
                                      return actual == expected;
                                  });
            });

        if(module_iter == submodule_list.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }

        auto* module_to_run = *module_iter;
        std::unordered_map<std::string, argument> p_map;

        // add input parameters to parameter_map
        auto in_param_names = get_input_parameter_names(module_to_run);
        assert(in_param_names.size() <= args.size());
        std::transform(in_param_names.begin(),
                       in_param_names.end(),
                       args.begin(),
                       std::inserter(p_map, p_map.end()),
                       [&](auto&& name, auto&& a) { return std::make_pair(name, a); });

        // Each output of the submodule writes into the caller's buffer for that output
        auto out_param_names = get_output_parameter_names(module_to_run);
        auto param_shapes    = module_to_run->get_parameter_shapes();
        auto module_outputs  = module_to_run->get_returns();
        if(not out_param_names.empty())
        {
            if(args.size() != in_param_names.size() + num_outputs())
                MIGRAPHX_THROW("SELECT_MODULE: missing output allocations");
            if(module_outputs.size() != num_outputs())
                MIGRAPHX_THROW(
                    "SELECT_MODULE: output allocation count does not match module outputs");
        }
        auto output_start = args.size() - num_outputs();
        for(const auto& name : out_param_names)
        {
            auto parameter = module_to_run->get_parameter(name);
            auto output    = std::find_if(
                module_outputs.begin(), module_outputs.end(), [&](instruction_ref result) {
                    return contains(instruction::get_output_alias(result), parameter);
                });
            if(output == module_outputs.end())
                MIGRAPHX_THROW("SELECT_MODULE: output parameter does not alias a module output");
            const auto& allocation =
                args.at(output_start + std::distance(module_outputs.begin(), output));
            auto ps = param_shapes.at(name);
            if(ps.bytes() > allocation.get_shape().bytes())
                MIGRAPHX_THROW("SELECT_MODULE: output allocation is too small");
            p_map.emplace(name, allocation.get_shape() == ps ? allocation : allocation.reshape(ps));
        }
        auto results = run(module_to_run, p_map);
        return argument{results};
    }

    // The caller's output buffers are appended after the input arguments during lowering.
    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        if(shapes.size() <= num_outputs())
            return {};
        std::vector<std::size_t> result(num_outputs());
        std::iota(result.begin(), result.end(), shapes.size() - num_outputs());
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
