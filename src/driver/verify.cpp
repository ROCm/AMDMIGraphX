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
#include "verify.hpp"
#include "perf.hpp"

#include <migraphx/algorithm.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/fp_to_double.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/verify_args.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/optional.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Gives tolerances based on user input (`rms_tol`, `atol`, `rtol` parameters) and defaults.
 * Sets to fp4 tolerances if any fp4x2_type is found.
 * Else sets to fp16 tolerances if `quantize` input is fp16 or any fp16 instruction is found in the
 * model.
 */
verify::tolerance get_tolerances(const program& p,
                                 const verify_options& vo,
                                 std::optional<double> rms_tol,
                                 std::optional<double> atol,
                                 std::optional<double> rtol)
{
    bool has_16bit = any_of(p.get_modules(), [](auto&& m) {
        return any_of(*m, [](auto&& ins) {
            return (ins.get_shape().type() == shape::half_type or
                    ins.get_shape().type() == shape::bf16_type);
        });
    });
    bool has_fp4   = any_of(p.get_modules(), [](auto&& m) {
        return any_of(*m, [](auto&& ins) { return (ins.get_shape().type() == shape::fp4x2_type); });
    });
    migraphx::verify::tolerance result{};
    if(has_fp4)
    {
        result.rms_tol = 8e-1;
        result.atol    = 4e-1;
        result.rtol    = 4e-1;
    }
    else if(has_16bit or vo.quantize == precision::fp16 or vo.quantize == precision::bf16)
    {
        result.rms_tol = 8e-2;
        result.atol    = 4e-2;
        result.rtol    = 4e-2;
    }
    if(rms_tol)
    {
        result.rms_tol = *rms_tol;
    }
    if(atol)
    {
        result.atol = *atol;
    }
    if(rtol)
    {
        result.rtol = *rtol;
    }
    return result;
}

namespace {

using trace_function      = std::function<void(instruction_ref, const argument&)>;
using substitute_function = std::function<optional<argument>(instruction_ref, const argument&)>;

// Captures ref outputs by debug symbol and compares each target op at its terminal symbol.
struct verify_callback
{
    struct layer_result
    {
        std::string symbol = {};
        std::string op     = {};
        std::size_t order  = 0;
        double rms_error   = 0;
        bool passed        = false;
    };

    struct ref_output
    {
        argument output   = {};
        std::size_t order = 0;
    };

    using ref_map = std::unordered_map<std::string, ref_output>;

    verify::tolerance tols = {};

    std::size_t ref_count                                 = 0;
    ref_map ref_outputs                                   = {};
    std::unordered_map<std::string, layer_result> results = {};

    trace_function capture()
    {
        return [this](instruction_ref ins, const argument& output) {
            auto order = ref_count++;
            auto buf   = output.share();
            for(const auto& symbol : ins->get_debug_symbols())
                ref_outputs[symbol] = {buf, order};
        };
    }

    // A fused op carries every symbol it absorbed; the last one traced is the output it produces.
    ref_map::const_iterator terminal(instruction_ref ins) const
    {
        auto result = ref_outputs.end();
        for(const auto& s : ins->get_debug_symbols())
        {
            auto it = ref_outputs.find(s);
            if(it == ref_outputs.end())
                continue;
            if(result == ref_outputs.end() or it->second.order > result->second.order)
                result = it;
        }
        return result;
    }

    // Scores the op, then feeds the reference forward so later ops run on known-good inputs and
    // each error is the op's own.
    substitute_function compare()
    {
        return [this](instruction_ref ins, const argument& output) -> optional<argument> {
            // Constants carry the node's symbol but are not its output.
            if(ins->can_eval())
                return nullopt;
            auto it = terminal(ins);
            if(it == ref_outputs.end())
                return nullopt;
            const auto& ref = it->second;
            // Reshapes and slices inherit the symbol; only the op whose lens match produced it.
            if(not shape::same_lens(ref.output.get_shape(), output.get_shape()))
                return nullopt;
            // The reference can differ in type (--ref-use-double) and layout, so restate it in the
            // target's shape; fill copies in logical order and converts.
            argument ref_arg{output.get_shape()};
            ref.output.visit([&](auto s) { ref_arg.fill(s.begin(), s.end()); });
            double rms  = 0;
            bool passed = false;
            visit_all(output, ref_arg)([&](auto t, auto r) {
                passed = verify::verify_range_with_tolerance(t, verify::expected{r}, tols, &rms);
            });
            // NaN never compares greater, so rank it worst.
            if(std::isnan(rms))
                rms = std::numeric_limits<double>::infinity();
            results[it->first] = {it->first, ins->name(), ref.order, rms, passed};
            return ref_arg;
        };
    }

    // Failing layers in reference execution order.
    std::vector<layer_result> failures() const
    {
        std::vector<layer_result> result;
        transform_if(
            results.begin(),
            results.end(),
            std::back_inserter(result),
            [](const auto& r) { return not r.second.passed; },
            [](const auto& r) { return r.second; });
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            return a.order < b.order;
        });
        return result;
    }

    // The failing layer with the largest error of its own.
    optional<layer_result> divergence_source() const
    {
        auto failed = failures();
        if(failed.empty())
            return nullopt;
        return *std::max_element(failed.begin(), failed.end(), [](const auto& a, const auto& b) {
            return a.rms_error < b.rms_error;
        });
    }
};

} // namespace

static std::vector<argument> run_ref(program p,
                                     const compile_options& options,
                                     const verify_options& vo,
                                     const parameter_map& inputs,
                                     trace_function trace = nullptr)
{
    if(vo.ref_use_double)
    {
        run_passes(
            p, {fp_to_double{}, simplify_qdq{.remove_qdq_only = true}, dead_code_elimination{}});
    }
    p.compile(migraphx::make_target("ref"), options);
    execution_environment exec_env{};
    exec_env.trace = std::move(trace);
    auto out       = p.eval(inputs, exec_env);
    log::info() << p;
    return out;
}

static std::vector<argument> run_target(program p,
                                        const target& t,
                                        const compile_options& options,
                                        const verify_options& vo,
                                        const parameter_map& inputs,
                                        substitute_function substitute = nullptr)
{
    if(vo.compiled_model.empty())
    {
        if(vo.quantize == precision::fp16)
        {
            quantize_fp16(p);
        }
        if(vo.quantize == precision::bf16)
        {
            quantize_bf16(p);
        }
        p.compile(t, options);
    }
    else
    {
        p = load(vo.compiled_model);
    }

    parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        auto arg   = inputs.count(x.first) == 0 ? generate_argument(x.second) : inputs.at(x.first);
        m[x.first] = options.offload_copy ? arg : t.copy_to(arg);
    }
    execution_environment exec_env{};
    exec_env.substitute = std::move(substitute);
    auto gpu_out        = p.eval(m, exec_env);
    std::vector<argument> output(gpu_out.size());
    log::info() << p;
    std::transform(gpu_out.begin(), gpu_out.end(), output.begin(), [&](auto& argu) {
        return options.offload_copy ? argu : t.copy_from(argu);
    });
    return output;
}

// Runs ref and target once each; nullopt when the model has no debug symbols to compare by.
static optional<verify_callback> run_layerwise_compare(const program& p,
                                                       const target& t,
                                                       const compile_options& options,
                                                       const verify_options& vo,
                                                       const parameter_map& inputs,
                                                       verify::tolerance tols)
{
    if(not any_of(p.get_modules(), [](auto* m) { return m->has_debug_symbols(); }))
    {
        log::error() << "Layer-wise comparison (--no-rebuild) requires debug symbols; reload the "
                        "model with --debug-symbols.";
        return nullopt;
    }
    verify_callback vcb{};
    vcb.tols = tols;
    run_ref(p, options, vo, inputs, vcb.capture());
    run_target(p, t, options, vo, inputs, vcb.compare());
    log::info() << "Layers compared: " << vcb.results.size();
    return vcb;
}

bool verify_program(const std::string& name,
                    const program& p,
                    const target& t,
                    const compile_options& options,
                    const verify_options& vo,
                    const parameter_map& inputs,
                    verify::tolerance tols)
{
    auto ref_outs    = run_ref(p, options, vo, inputs);
    auto target_outs = run_target(p, t, options, vo, inputs);

    std::size_t output_num = ref_outs.size();
    bool passed            = true;
    for(std::size_t i = 0; i < output_num; ++i)
    {
        if(ref_outs[i].get_shape().type() != target_outs[i].get_shape().type() or
           ref_outs[i].get_shape().lens() != target_outs[i].get_shape().lens())
        {
            log::error() << "FAILED: " << name;
            log::error() << "Shape mismatch {" << ref_outs[i].get_shape() << "} != {"
                         << target_outs[i].get_shape() << "}";
        }
        else
        {
            passed &= verify_args(name, target_outs[i], verify::expected{ref_outs[i]}, tols);
        }
    }
    if(passed)
        log::info() << "MIGraphX verification passed successfully.";
    return passed;
}

void verify_instructions(const program& prog,
                         const target& t,
                         const compile_options& options,
                         const verify_options& vo,
                         verify::tolerance tols)
{
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
            verify_program(ins.name(), p, t, options, vo, create_param_map(p, false), tols);
        }
        catch(...)
        {
            log::error() << "Instruction " << ins.name() << " threw an exception.";
            throw;
        }
    }
}

static bool verify_reduced(program p,
                           int n,
                           const target& t,
                           const compile_options& options,
                           const verify_options& vo,
                           const parameter_map& inputs,
                           verify::tolerance tols)
{
    auto* mm  = p.get_main_module();
    auto last = std::prev(mm->end(), n);
    mm->remove_instructions(last, mm->end());
    log::info() << "Verify: " << n;
    log::info() << p;
    try
    {
        return verify_program(std::to_string(n), p, t, options, vo, inputs, tols);
    }
    catch(const std::exception& e)
    {
        log::error() << "FAILED: " << n;
        log::error() << "Exception: " << e.what();
        return false;
    }
}

void verify_reduced_program(const program& p,
                            const target& t,
                            const compile_options& options,
                            const verify_options& vo,
                            const parameter_map& inputs,
                            verify::tolerance tols)
{
    if(vo.no_rebuild)
    {
        auto vcb = run_layerwise_compare(p, t, options, vo, inputs, tols);
        if(not vcb)
            return;
        auto failures = vcb->failures();
        for(const auto& lr : failures)
            log::error() << "FAILED at " << lr.symbol << " (" << lr.op << ")";
        if(failures.empty())
            log::info() << "MIGraphX verification passed successfully.";
        return;
    }

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
        verify_reduced(p, i, t, options, vo, inputs, tols);
    }
}

static std::unordered_map<instruction_ref, std::size_t> accumulate_weights(instruction_ref last)
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

static optional<instruction_ref>
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

static std::vector<std::size_t> find_trim_instructions(const module& m)
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

void verify_bisected_program(const program& p,
                             const target& t,
                             const compile_options& options,
                             const verify_options& vo,
                             const parameter_map& inputs,
                             verify::tolerance tols)
{
    // Reports the source layer by symbol, unlike the bisect loop below which reports the first
    // failing trim count.
    if(vo.no_rebuild)
    {
        auto vcb = run_layerwise_compare(p, t, options, vo, inputs, tols);
        if(not vcb)
            return;
        auto failure = vcb->divergence_source();
        if(failure)
            std::cout << "Failure introduced at: " << failure->symbol << " (" << failure->op << ")"
                      << std::endl;
        else
            log::info() << "MIGraphX verification passed successfully.";
        return;
    }

    const auto* mm = p.get_main_module();

    std::vector<std::size_t> trims = find_trim_instructions(*mm);
    std::int64_t right             = trims.size();
    std::int64_t left              = 0;
    std::int64_t failed            = -1;

    while(left <= right)
    {
        std::int64_t mid = left + (right - left) / 2;
        assert(mid < trims.size() and mid >= 0);
        std::int64_t trim = trims.rbegin()[mid];
        bool passed       = verify_reduced(p, trim, t, options, vo, inputs, tols);
        if(passed)
        {
            left = mid + 1;
        }
        else
        {
            failed = trim;
            right  = mid - 1;
        }
    }
    if(failed > 0)
    {
        std::cout << "Failure starts at: " << failed << std::endl;
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
