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
#include <migraphx/config.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/symbolic_shape_helpers.hpp>
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct select_module
{
    // output_dyn_shapes describes the output shapes returned by candidates.
    shape output_dyn_shapes;
    // logical_output_dyn_shapes describes the actual unpadded output shapes.
    // These are different from output_dyn_shapes when bucketizing adds fixed padding.
    optional<shape> logical_output_dyn_shapes;
    // Captures symbolic input expressions used to evaluate logical output shapes at runtime.
    std::vector<shape> input_shapes;

    struct output_location
    {
        std::size_t index;
        std::vector<std::size_t> path;
    };

    struct parameter_metadata
    {
        std::string name;
        shape parameter_shape;
        // Caller buffers this parameter writes, in parameter-leaf order. A location identifies
        // both the submodule return slot and a path within that return's buffer.
        std::vector<output_location> output_locations;
    };

    struct module_metadata
    {
        std::size_t module_index;
        std::vector<parameter_metadata> inputs;
        std::vector<parameter_metadata> outputs;
        std::vector<std::size_t> selector_indices;
        // Dynamic inputs that every candidate shares. Parameter evaluation does not check
        // dynamic shapes, so these are checked once the module is selected.
        std::vector<std::size_t> shared_dynamic_indices;
    };

    struct module_set_metadata
    {
        std::vector<module_metadata> modules;
    };

    // Built from the instruction's fixed submodule list during finalization. Candidate indices
    // avoid retaining non-owning module pointers when a finalized program is copied.
    module_set_metadata metadata;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_dyn_shapes, "output_dyn_shapes"),
                    f(self.logical_output_dyn_shapes, "logical_output_dyn_shapes"),
                    f(self.input_shapes, "input_shapes"));
    }

    std::string name() const { return "select_module"; }

    std::size_t num_outputs() const { return output_dyn_shapes.sub_shapes().size(); }

    static MIGRAPHX_EXPORT bool compatible_output_shape(const shape& logical, const shape& output);

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>&) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);
        if(not logical_output_dyn_shapes.has_value())
            return shape{output_dyn_shapes};
        if(logical_output_dyn_shapes->sub_shapes().size() != num_outputs())
            MIGRAPHX_THROW("SELECT_MODULE: logical output count does not match outputs");
        if(input_shapes.empty() or input_shapes.size() > inputs.size())
            MIGRAPHX_THROW("SELECT_MODULE: symbolic input shapes were not captured");

        // Check that each logical output shape is compatible with its physical output shape.
        migraphx::for_each(logical_output_dyn_shapes->sub_shapes().begin(),
                           logical_output_dyn_shapes->sub_shapes().end(),
                           output_dyn_shapes.sub_shapes().begin(),
                           [&](const shape& logical, const shape& output) {
                               if(not compatible_output_shape(logical, output))
                                   MIGRAPHX_THROW(
                                       "SELECT_MODULE: logical output is incompatible with "
                                       "declared output");
                           });

        std::vector<sym::expr> expressions;
        fix([&](auto self, const shape& s) {
            if(s.type() == shape::tuple_type)
            {
                std::for_each(s.sub_shapes().begin(),
                              s.sub_shapes().end(),
                              [&](const shape& sub_shape) { self(sub_shape); });
                return;
            }
            if(s.dynamic() and not s.symbolic())
                MIGRAPHX_THROW("SELECT_MODULE: logical outputs must be static or symbolic");
            if(not s.symbolic())
                return;
            std::transform(s.dyn_dims().begin(),
                           s.dyn_dims().end(),
                           std::back_inserter(expressions),
                           [](const auto& dim) { return dim.sym_expr; });
            expressions.insert(expressions.end(), s.dyn_strides().begin(), s.dyn_strides().end());
        })(*logical_output_dyn_shapes);
        if(auto unresolved = find_unresolved_shape_expression(input_shapes, expressions))
            MIGRAPHX_THROW("SELECT_MODULE: logical output expression '" + unresolved->to_string() +
                           "' cannot be resolved from the input dimensions");
        return shape{output_dyn_shapes};
    }

    void finalize(context&,
                  const shape&,
                  const std::vector<shape>& inputs,
                  const std::vector<module_ref>& submodule_list)
    {
        if(logical_output_dyn_shapes.has_value() and
           (input_shapes.empty() or input_shapes.size() > inputs.size()))
            MIGRAPHX_THROW("SELECT_MODULE: symbolic input shapes were not captured");
        metadata = build_module_metadata(submodule_list);
    }

    std::vector<shape> compute_logical_output_shapes(const std::vector<argument>& args) const
    {
        if(not logical_output_dyn_shapes.has_value())
            return {};
        if(input_shapes.empty() or input_shapes.size() > args.size())
            MIGRAPHX_THROW("SELECT_MODULE: symbolic input shapes were not captured");

        std::unordered_map<sym::expr, std::size_t> values;
        migraphx::for_each(
            input_shapes.begin(),
            input_shapes.end(),
            args.begin(),
            [&](const shape& expected, const argument& arg) {
                const auto& actual = arg.get_shape();
                if(not compatible_input_shape(actual, expected))
                    MIGRAPHX_THROW(
                        "SELECT_MODULE: runtime input does not match symbolic input shape");
                if(not expected.symbolic())
                    return;
                if(expected.ndim() != actual.ndim())
                    MIGRAPHX_THROW(
                        "SELECT_MODULE: runtime input rank does not match symbolic input");
                migraphx::for_each(expected.dyn_dims().begin(),
                                   expected.dyn_dims().end(),
                                   actual.lens().begin(),
                                   [&](const auto& dim, auto len) {
                                       if(not dim.is_symbolic())
                                           return;
                                       auto expression = sym::as_symbol(dim.sym_expr);
                                       auto result     = values.emplace(expression, len);
                                       if(not result.second and result.first->second != len)
                                           MIGRAPHX_THROW("SELECT_MODULE: repeated expression has "
                                                          "inconsistent runtime dimensions");
                                   });
            });

        std::vector<shape> result;
        std::transform(logical_output_dyn_shapes->sub_shapes().begin(),
                       logical_output_dyn_shapes->sub_shapes().end(),
                       std::back_inserter(result),
                       [&](const shape& s) { return s.to_static(values); });
        return result;
    }

    std::vector<std::string> get_input_parameter_names(const_module_ref mod) const
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

    std::vector<std::string> get_output_parameter_names(const_module_ref mod) const
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

    static bool compatible_input_shape(const shape& actual, const shape& expected)
    {
        if(actual == expected)
            return true;
        if(expected.dynamic())
            return actual.type() == expected.type() and shape::is_compatible_lens(actual, expected);
        if(actual.type() != expected.type() or actual.type() == shape::tuple_type or
           not shape::same_lens(actual, expected))
            return false;
        if(actual.elements() == 0)
            return true;
        return all_of(range(actual.ndim()), [&](auto axis) {
            return actual.lens().at(axis) <= 1 or
                   actual.strides().at(axis) == expected.strides().at(axis);
        });
    }

    MIGRAPHX_EXPORT module_set_metadata
    build_module_metadata(const std::vector<module_ref>& candidates) const;

    const module_set_metadata& finalized_metadata(std::size_t module_count) const
    {
        if(metadata.modules.empty() or metadata.modules.size() != module_count)
            MIGRAPHX_THROW("SELECT_MODULE: operation was not finalized");
        return metadata;
    }

    static bool empty_output_shape(const shape& s)
    {
        if(s.type() != shape::tuple_type)
            return s.elements() == 0;
        return all_of(s.sub_shapes(),
                      [](const shape& sub_shape) { return empty_output_shape(sub_shape); });
    }

    argument short_circuit_outputs(const module_set_metadata& finalized,
                                   const std::vector<argument>& args,
                                   const std::vector<shape>& output_shapes) const
    {
        std::vector<argument> results;
        results.reserve(num_outputs());
        if(not finalized.modules.front().outputs.empty())
        {
            if(args.size() < num_outputs())
                MIGRAPHX_THROW("SELECT_MODULE: missing output allocations");
            auto output_args = args.end() - num_outputs();
            migraphx::for_each(
                output_args,
                args.end(),
                output_shapes.begin(),
                [&](const argument& arg, const shape& s) {
                    results.push_back(fix<argument>(
                        [&](auto self, const argument& current, const shape& output_shape) {
                            if(output_shape.type() != shape::tuple_type)
                            {
                                auto empty_shape = shape{output_shape.type(),
                                                         output_shape.lens(),
                                                         current.get_shape().strides()};
                                return current.reshape(empty_shape);
                            }
                            auto sub_args = current.get_sub_objects();
                            if(sub_args.size() != output_shape.sub_shapes().size())
                                MIGRAPHX_THROW(
                                    "SELECT_MODULE: output tuple does not match logical output");
                            std::vector<argument> sub_results;
                            migraphx::for_each(
                                sub_args.begin(),
                                sub_args.end(),
                                output_shape.sub_shapes().begin(),
                                [&](const argument& sub_arg, const shape& sub_shape) {
                                    sub_results.push_back(self(sub_arg, sub_shape));
                                });
                            return argument{sub_results};
                        })(arg, s));
                });
        }
        else
        {
            migraphx::for_each(
                output_shapes.begin(),
                output_shapes.end(),
                output_dyn_shapes.sub_shapes().begin(),
                [&](const shape& logical, const shape& physical) {
                    results.push_back(fix<argument>([](auto self, const shape& x, const shape& y) {
                        if(x.type() != shape::tuple_type)
                        {
                            auto physical_shape = y.dynamic() ? shape{y.type(), y.max_lens()} : y;
                            auto empty_shape = shape{x.type(), x.lens(), physical_shape.strides()};
                            return argument{empty_shape, nullptr};
                        }
                        std::vector<argument> sub_objects;
                        migraphx::for_each(x.sub_shapes().begin(),
                                           x.sub_shapes().end(),
                                           y.sub_shapes().begin(),
                                           [&](const shape& xsub, const shape& ysub) {
                                               sub_objects.push_back(self(xsub, ysub));
                                           });
                        return argument{sub_objects};
                    })(logical, physical));
                });
        }
        return argument{results};
    }

    // Input arguments are ordered like the sorted input parameters, followed by one buffer per
    // output when the submodules have output parameters. Only the positions whose parameter
    // differs between candidates are compared to pick a module. A shared input that does not match
    // would reject every candidate, so the shared dynamic inputs are only checked on the selected
    // module; the static ones are checked when the submodule evaluates its parameters.
    template <class GetArgument>
    const module_metadata& find_module(const module_set_metadata& modules_metadata,
                                       std::size_t argument_count,
                                       GetArgument get_argument) const
    {
        auto module_iter =
            std::find_if(modules_metadata.modules.begin(),
                         modules_metadata.modules.end(),
                         [&](const auto& info) {
                             return info.inputs.size() <= argument_count and
                                    std::all_of(info.selector_indices.begin(),
                                                info.selector_indices.end(),
                                                [&](std::size_t index) {
                                                    return compatible_input_shape(
                                                        get_argument(index).get_shape(),
                                                        info.inputs[index].parameter_shape);
                                                });
                         });

        if(module_iter == modules_metadata.modules.end() or
           not std::all_of(module_iter->shared_dynamic_indices.begin(),
                           module_iter->shared_dynamic_indices.end(),
                           [&](std::size_t index) {
                               return compatible_input_shape(
                                   get_argument(index).get_shape(),
                                   module_iter->inputs[index].parameter_shape);
                           }))
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }
        if(not module_iter->outputs.empty() and
           argument_count != module_iter->inputs.size() + num_outputs())
            MIGRAPHX_THROW("SELECT_MODULE: missing output allocations");
        return *module_iter;
    }

    argument prepare_output_shape(const parameter_metadata& output,
                                  const shape& expected,
                                  const argument& arg) const
    {
        if(arg.get_shape() == expected)
            return arg;
        // Reshaping onto a smaller buffer would let the submodule write past its end, so refuse
        // rather than corrupt memory.
        if(arg.get_shape().bytes() < expected.bytes())
            MIGRAPHX_THROW("SELECT_MODULE: output buffer for \"" + output.name + "\" holds " +
                           std::to_string(arg.get_shape().bytes()) + " bytes but the selected " +
                           "submodule writes " + std::to_string(expected.bytes()));
        return arg.reshape(expected);
    }

    // get_output returns the caller's buffer for a return of the submodule.
    template <class GetOutput>
    argument prepare_output(const parameter_metadata& output, GetOutput get_output) const
    {
        std::size_t leaf_index = 0;
        auto prepared          = fix<argument>([&](auto self, const shape& expected) {
            if(expected.type() != shape::tuple_type)
            {
                const auto& location = output.output_locations.at(leaf_index++);
                auto result          = get_output(location.index);
                for(auto index : location.path)
                    result = result.get_sub_object(index);
                return prepare_output_shape(output, expected, result);
            }

            std::vector<argument> result;
            result.reserve(expected.sub_shapes().size());
            std::transform(expected.sub_shapes().begin(),
                           expected.sub_shapes().end(),
                           std::back_inserter(result),
                           [&](const shape& sub_shape) { return self(sub_shape); });
            return argument{result};
        })(output.parameter_shape);
        assert(leaf_index == output.output_locations.size());
        return prepared;
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        const auto& finalized = finalized_metadata(submodule_list.size());
        auto logical_outputs  = compute_logical_output_shapes(args);
        if(not logical_outputs.empty() and
           all_of(logical_outputs, [](const shape& s) { return empty_output_shape(s); }))
            return short_circuit_outputs(finalized, args, logical_outputs);
        const auto& module_info =
            find_module(finalized, args.size(), [&](std::size_t index) -> const argument& {
                return args[index];
            });
        auto* module_to_run = submodule_list.at(module_info.module_index);
        std::unordered_map<std::string, argument> p_map;
        p_map.reserve(module_info.inputs.size() + module_info.outputs.size());

        // add input parameters to parameter_map
        std::transform(module_info.inputs.begin(),
                       module_info.inputs.end(),
                       args.begin(),
                       std::inserter(p_map, p_map.end()),
                       [](const auto& input, const auto& arg) {
                           const auto& expected = input.parameter_shape;
                           if(not expected.dynamic() and arg.get_shape() != expected)
                           {
                               if(not compatible_input_shape(arg.get_shape(), expected))
                                   MIGRAPHX_THROW(
                                       "SELECT_MODULE: no compatible submodules found for given "
                                       "input shapes");
                               return std::make_pair(input.name, arg.reshape(expected));
                           }
                           return std::make_pair(input.name, arg);
                       });

        // Each output of the submodule writes into the caller's buffer for that output. A compiled
        // output parameter can itself be a tuple when one kernel produces multiple returns.
        auto output_start = args.size() - num_outputs();
        auto get_output   = [&](std::size_t index) { return args[output_start + index]; };
        std::transform(module_info.outputs.begin(),
                       module_info.outputs.end(),
                       std::inserter(p_map, p_map.end()),
                       [&](const auto& output) {
                           return std::make_pair(output.name, prepare_output(output, get_output));
                       });
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
