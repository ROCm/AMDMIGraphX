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

#include <migraphx/program_verify.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/execution_environment.hpp>
#include <migraphx/fp_to_double.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/verify_args.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace verify {

namespace {

using trace_function      = std::function<void(instruction_ref, const argument&)>;
using substitute_function = std::function<optional<argument>(instruction_ref, const argument&)>;

parameter_map make_inputs(const program& p, const parameter_map& inputs)
{
    parameter_map result = inputs;
    auto shapes          = p.get_parameter_shapes();
    transform_if(
        shapes.begin(),
        shapes.end(),
        std::inserter(result, result.end()),
        [&](const auto& item) { return not contains(inputs, item.first); },
        [](const auto& item) {
            return std::make_pair(item.first, generate_argument(item.second));
        });
    return result;
}

std::vector<argument> run_ref(program p,
                              const compile_options& options,
                              bool ref_use_double,
                              const parameter_map& inputs,
                              trace_function trace = nullptr)
{
    if(ref_use_double)
    {
        run_passes(
            p, {fp_to_double{}, simplify_qdq{.remove_qdq_only = true}, dead_code_elimination{}});
    }
    p.compile(make_target("ref"), options);
    execution_environment exec_env{};
    exec_env.trace = std::move(trace);
    auto out       = p.eval(inputs, exec_env);
    log::info() << p;
    return out;
}

std::vector<argument> run_target(program p,
                                 const target& t,
                                 const program_options& options,
                                 const parameter_map& inputs,
                                 substitute_function substitute = nullptr)
{
    if(options.compiled_model.empty())
    {
        if(options.quantize == program_precision::fp16)
        {
            quantize_fp16(p);
        }
        if(options.quantize == program_precision::bf16)
        {
            quantize_bf16(p);
        }
        p.compile(t, options.compile);
    }
    else
    {
        p = load(options.compiled_model);
    }

    parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        auto arg   = inputs.count(x.first) == 0 ? generate_argument(x.second) : inputs.at(x.first);
        m[x.first] = options.compile.offload_copy ? arg : t.copy_to(arg);
    }
    execution_environment exec_env{};
    exec_env.substitute = std::move(substitute);
    auto gpu_out        = p.eval(m, exec_env);
    std::vector<argument> output(gpu_out.size());
    log::info() << p;
    std::transform(gpu_out.begin(), gpu_out.end(), output.begin(), [&](auto& argu) {
        return options.compile.offload_copy ? argu : t.copy_from(argu);
    });
    return output;
}

std::string source_name(instruction_ref ins, const std::string& label)
{
    const auto& symbols = ins->get_debug_symbols();
    std::vector<std::string> names;
    std::copy_if(symbols.begin(), symbols.end(), std::back_inserter(names), [](const auto& symbol) {
        return not starts_with(symbol, "@verify:");
    });
    if(names.empty())
        return "#" + remove_prefix(label, "@verify:");
    return join_strings(std::move(names), ", ");
}

struct verify_callback
{
    struct ref_output
    {
        argument output   = {};
        std::string name  = {};
        std::size_t order = 0;
    };

    using ref_map = std::unordered_map<std::string, ref_output>;

    tolerance tols = {};

    std::size_t ref_count                       = 0;
    ref_map ref_outputs                         = {};
    std::map<std::size_t, layer_result> results = {};

    std::vector<instruction_ref> source_instructions = {};

    // Captures ref outputs for each instruction.
    trace_function capture()
    {
        return [this](instruction_ref ins, const argument& output) {
            if(output.get_shape().type() == shape::tuple_type)
                return;
            auto order = ref_count++;
            for(const auto& symbol : ins->get_debug_symbols())
                if(starts_with(symbol, "@verify:"))
                    ref_outputs[symbol] = {output, source_name(ins, symbol), order};
        };
    }

    // Returns the terminal reference output when compatible symbols form a chain.
    ref_map::const_iterator terminal(instruction_ref ins, const shape& s) const
    {
        std::vector<ref_map::const_iterator> matches;
        for(const auto& symbol : ins->get_debug_symbols())
        {
            auto it = ref_outputs.find(symbol);
            if(it == ref_outputs.end())
                continue;
            const auto& rs = it->second.output.get_shape();
            // quantization changes the type, so only check for float vs integer
            if(not shape::same_lens(rs, s) or
               shape::is_integral(rs.type()) != shape::is_integral(s.type()))
                continue;
            matches.push_back(it);
        }
        auto result = std::max_element(matches.begin(), matches.end(), [](auto x, auto y) {
            return x->second.order < y->second.order;
        });
        if(result == matches.end())
            return ref_outputs.end();
        auto source = [&](auto x) {
            return source_instructions.at(std::stoull(x->first.substr(x->first.find(':') + 1)));
        };
        if(any_of(matches, [&](auto other) {
               return other->second.order != (*result)->second.order and
                      not reaches(source(other), source(*result));
           }))
            return ref_outputs.end();
        return *result;
    }

    // Scores the target output against the captured ref output, then returns the ref value so
    // later layers read known-good inputs and each error is the layer's own.
    substitute_function compare()
    {
        return [this](instruction_ref ins, const argument& output) -> optional<argument> {
            if(ins->can_eval() or ends_with(ins->name(), "::literal") or
               contains({"broadcast", "multibroadcast"}, ins->name()))
                return nullopt;
            if(output.get_shape().type() == shape::tuple_type)
                return nullopt;
            auto it = terminal(ins, output.get_shape());
            if(it == ref_outputs.end())
                return nullopt;
            const auto& ref = it->second;
            assert(ref.output.get_shape().elements() == output.get_shape().elements());
            auto ref_arg = ref.output;
            if(ref.output.get_shape() != output.get_shape())
            {
                ref_arg = argument{output.get_shape()};
                ref.output.visit([&](auto s) { ref_arg.fill(s.begin(), s.end()); });
            }
            double rms  = 0;
            bool passed = false;
            visit_all(output, ref_arg)([&](auto t, auto r) {
                passed = verify_range_with_tolerance(t, expected{r}, tols, &rms);
            });
            // NaN never compares greater, so rank it worst.
            if(std::isnan(rms))
                rms = std::numeric_limits<double>::infinity();
            auto op            = ins->get_operator().attributes().get("group", ins->name());
            results[ref.order] = {
                .name = ref.name, .op = op, .index = ref.order, .rms_error = rms, .passed = passed};
            return ref_arg;
        };
    }
};

program label_instructions(program p)
{
    std::size_t id = 0;
    auto* m        = p.get_main_module();
    for(auto ins : iterator_for(*m))
    {
        if(ins->name() == "@return")
            continue;
        m->add_debug_symbols(ins, {"@verify:" + std::to_string(id++)});
    }
    return p;
}

parameter_map make_instruction_inputs(const program& p)
{
    parameter_map inputs;
    for(auto&& x : p.get_parameter_shapes())
        inputs[x.first] =
            generate_argument(x.second, std::hash<std::string>{}(x.first), random_mode::random);
    return inputs;
}

program_result verify_outputs(const program& p,
                              const target& t,
                              const parameter_map& inputs,
                              const program_options& options,
                              const std::string& name)
{
    auto ref_outs    = run_ref(p, options.compile, options.ref_use_double, inputs);
    auto target_outs = run_target(p, t, options, inputs);

    program_result result;
    if(ref_outs.size() != target_outs.size())
    {
        auto message = "Output count mismatch {" + std::to_string(ref_outs.size()) + "} != {" +
                       std::to_string(target_outs.size()) + "}";
        log::error() << "FAILED: " << name;
        log::error() << message;
        result.success = false;
        result.results.push_back({.name = name, .op = "@return", .message = std::move(message)});
        return result;
    }

    std::size_t output_num = ref_outs.size();
    bool passed            = true;
    for(std::size_t i = 0; i < output_num; ++i)
    {
        layer_result layer{.name = name, .op = "@return", .index = i};
        if(ref_outs[i].get_shape().type() != target_outs[i].get_shape().type() or
           ref_outs[i].get_shape().lens() != target_outs[i].get_shape().lens())
        {
            layer.message = "Shape mismatch {" + to_string(ref_outs[i].get_shape()) + "} != {" +
                            to_string(target_outs[i].get_shape()) + "}";
            log::error() << "FAILED: " << name;
            log::error() << layer.message;
        }
        else
        {
            layer.passed = verify_args(
                name, target_outs[i], expected{ref_outs[i]}, options.tols, &layer.rms_error);
        }
        passed &= layer.passed;
        result.results.push_back(std::move(layer));
    }
    if(passed)
        log::info() << "MIGraphX verification passed successfully.";
    result.success = passed;
    return result;
}

program_result
verify_instructions(const program& prog, const target& t, const program_options& options)
{
    program_result result;
    const auto* mm_prog = prog.get_main_module();
    for(auto&& ins : (*mm_prog))
    {
        if(ins.name().front() == '@')
            continue;
        if(ins.name() == "broadcast")
            continue;
        if(ins.name() == "transpose")
            continue;
        if(ins.name() == "reshape")
            continue;
        if(ins.name() == "undefined")
            continue;
        program p;
        auto* mm_p = p.get_main_module();
        std::vector<instruction_ref> inputs;
        for(auto&& arg : ins.inputs())
        {
            if(arg->name() == "@literal")
                inputs.push_back(mm_p->add_literal(arg->get_literal()));
            else
                inputs.push_back(
                    mm_p->add_parameter(std::to_string(inputs.size()), arg->get_shape()));
        }
        mm_p->add_instruction(ins.get_operator(), inputs);
        try
        {
            log::info() << "Verify: " << ins.name();
            std::cout << p << std::endl;
            auto verification =
                verify_outputs(p, t, make_instruction_inputs(p), options, ins.name());
            result.success = result.success and verification.success;
            result.results.insert(result.results.end(),
                                  std::make_move_iterator(verification.results.begin()),
                                  std::make_move_iterator(verification.results.end()));
        }
        catch(...)
        {
            log::error() << "Instruction " << ins.name() << " threw an exception.";
            throw;
        }
    }
    return result;
}

program_result verify_reduced(program p,
                              std::size_t n,
                              const target& t,
                              const program_options& options,
                              const parameter_map& inputs)
{
    auto* mm  = p.get_main_module();
    auto last = std::prev(mm->end(), n);
    mm->remove_instructions(last, mm->end());
    log::info() << "Verify: " << n;
    log::info() << p;
    try
    {
        auto name   = std::to_string(n);
        auto result = verify_outputs(p, t, inputs, options, name);
        for(auto& layer : result.results)
            layer.index = n;
        return result;
    }
    catch(const std::exception& e)
    {
        log::error() << "FAILED: " << n;
        log::error() << "Exception: " << e.what();
        program_result result;
        result.success = false;
        result.results.push_back(
            {.name = std::to_string(n), .message = e.what(), .index = n, .exception = true});
        return result;
    }
}

program_result verify_reduced_program(const program& p,
                                      const target& t,
                                      const program_options& options,
                                      const parameter_map& inputs)
{
    program_result result;
    const auto* mm = p.get_main_module();
    auto n         = std::distance(mm->begin(), mm->end());
    log::info() << "Verify steps: " << n;
    for(std::size_t i = 1; i < n; i++)
    {
        auto last = std::prev(mm->end(), i + 1);
        if(contains({"@literal", "@param"}, last->name()))
        {
            log::info() << "Skip: " << i;
            continue;
        }
        auto verification = verify_reduced(p, i, t, options, inputs);
        result.success    = result.success and verification.success;
        result.results.insert(result.results.end(),
                              std::make_move_iterator(verification.results.begin()),
                              std::make_move_iterator(verification.results.end()));
    }
    return result;
}

std::unordered_map<instruction_ref, std::size_t> accumulate_weights(instruction_ref last)
{
    std::unordered_map<instruction_ref, std::size_t> weights;
    fix<std::size_t>([&](auto self, auto ins) -> std::size_t {
        if(not contains(weights, ins))
        {
            if(ins->can_eval())
                return 0;
            std::size_t weight = 1;
            weights[ins]       = std::accumulate(
                ins->inputs().begin(),
                ins->inputs().end(),
                weight,
                [&](std::size_t w, instruction_ref i) -> std::size_t { return w + self(i); });
        }
        return weights[ins];
    })(last);
    return weights;
}

optional<instruction_ref>
get_parent(const std::unordered_map<instruction_ref, std::size_t>& weights, instruction_ref ins)
{
    if(ins->inputs().empty())
        return nullopt;
    auto next = std::max_element(ins->inputs().begin(),
                                 ins->inputs().end(),
                                 by(std::less<>{}, [&](instruction_ref input) -> std::size_t {
                                     if(not contains(weights, input))
                                         return 0;
                                     return weights.at(input);
                                 }));
    return *next;
}

std::vector<std::size_t> find_trim_instructions(const module& m)
{
    std::vector<std::size_t> result;
    auto last     = std::prev(m.end());
    auto weights  = accumulate_weights(last);
    auto next     = get_parent(weights, last);
    std::size_t i = 0;
    while(auto parent = get_parent(weights, *next))
    {
        i += std::distance(*parent, *next);
        result.push_back(i + 1);
        next = parent;
    }
    return result;
}

program_result verify_bisected(const program& p,
                               const target& t,
                               const program_options& options,
                               const parameter_map& inputs)
{
    const auto* mm = p.get_main_module();

    std::vector<std::size_t> trims = find_trim_instructions(*mm);
    std::int64_t right             = static_cast<std::int64_t>(trims.size()) - 1;
    std::int64_t left              = 0;
    program_result result;

    while(left <= right)
    {
        std::int64_t mid = left + (right - left) / 2;
        assert(mid < trims.size() and mid >= 0);
        auto trim         = trims.rbegin()[mid];
        auto verification = verify_reduced(p, trim, t, options, inputs);
        result.results.insert(result.results.end(),
                              std::make_move_iterator(verification.results.begin()),
                              std::make_move_iterator(verification.results.end()));
        if(verification.success)
        {
            left = mid + 1;
        }
        else
        {
            result.failure_step = trim;
            right               = mid - 1;
        }
    }
    if(result.failure_step)
    {
        std::cout << "Failure starts at: " << *result.failure_step << std::endl;
    }
    result.success = not result.failure_step.has_value();
    return result;
}

program_result verify_layerwise(const program& p,
                                const target& t,
                                const parameter_map& inputs,
                                const program_options& options)
{
    auto labeled = label_instructions(p);
    verify_callback vcb{options.tols};
    copy_if(iterator_for(*p.get_main_module()),
            std::back_inserter(vcb.source_instructions),
            [](auto ins) { return ins->name() != "@return"; });
    run_ref(labeled, options.compile, options.ref_use_double, inputs, vcb.capture());
    run_target(std::move(labeled), t, options, inputs, vcb.compare());
    program_result result;
    if(vcb.results.empty())
    {
        log::error() << "Layerwise comparison (--layerwise) matched no layers between the "
                        "reference and the target.";
        result.success = false;
        return result;
    }
    log::info() << "Layers compared: " << vcb.results.size();
    std::transform(vcb.results.begin(),
                   vcb.results.end(),
                   std::back_inserter(result.results),
                   [](auto& item) { return std::move(item.second); });
    auto failures = result.failures();
    if(failures.empty())
    {
        log::info() << "MIGraphX verification passed successfully.";
        return result;
    }
    for(const auto& lr : failures)
        log::error() << "FAILED at " << lr.name << " (" << lr.op << ")";
    auto source = std::max_element(failures.begin(),
                                   failures.end(),
                                   by(std::less<>{}, [](const auto& lr) { return lr.rms_error; }));
    std::cout << "Failure introduced at: " << source->name << " (" << source->op << ")"
              << std::endl;
    result.success = false;
    return result;
}

} // namespace

bool program_result::passed() const { return success; }

std::vector<layer_result> program_result::failures() const
{
    std::vector<layer_result> output;
    std::copy_if(results.begin(),
                 results.end(),
                 std::back_inserter(output),
                 [](const auto& result) { return not result.passed; });
    return output;
}

program_result verify_program(const program& p,
                              const target& t,
                              program_mode mode,
                              const parameter_map& inputs,
                              const program_options& options)
{
    if(mode != program_mode::outputs and not options.compiled_model.empty())
        MIGRAPHX_THROW("Compiled models are only supported for output verification.");
    program_result result;
    parameter_map values;
    if(mode != program_mode::instructions)
        values = make_inputs(p, inputs);
    switch(mode)
    {
    case program_mode::outputs: result = verify_outputs(p, t, values, options, options.name); break;
    case program_mode::instructions: result = verify_instructions(p, t, options); break;
    case program_mode::reduce: result = verify_reduced_program(p, t, options, values); break;
    case program_mode::bisect: result = verify_bisected(p, t, options, values); break;
    case program_mode::layerwise: result = verify_layerwise(p, t, values, options); break;
    }
    result.mode = mode;
    return result;
}

} // namespace verify
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
