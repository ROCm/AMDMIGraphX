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
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/fileutils.hpp>
#include <migraphx/json.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lower_device_ops.hpp>
#include <migraphx/gpu/time_op.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_COMPILE_PARALLEL);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_BENCHMARKING);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SKIP_BENCHMARKING);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_DUMP_BENCHMARK_MXR);

// Match the fixed per-candidate work budget used before adaptive timing.
constexpr std::size_t benchmark_samples = 20;

// Samples the coarse stage may spend on one candidate. Ranking a fast candidate from a single
// bundled run lets event overhead decide the order, and four is the smallest count that
// common_average trims. Slow candidates still collapse to fewer samples through min_samples.
constexpr std::size_t coarse_samples = 4;

adaptive_tuning_options compile_ops_tuning_overrides::resolve() const
{
    const auto apply = [](std::size_t& output,
                          const optional<std::int64_t>& input,
                          const char* name,
                          bool allow_zero) {
        if(not input.has_value())
            return;
        if(*input < 0)
            MIGRAPHX_THROW(std::string{name} + " must not be negative");
        if(not allow_zero and *input == 0)
            MIGRAPHX_THROW(std::string{name} + " must be greater than zero");
        output = static_cast<std::size_t>(*input);
    };

    adaptive_tuning_options result;
    apply(result.top_k, top_k, "tuning_top_k", true);
    apply(result.coarse.target_ms, coarse_target_ms, "tuning_coarse_target_ms", false);
    apply(result.precise.target_ms, precise_target_ms, "tuning_precise_target_ms", false);
    apply(result.coarse.max_samples, max_samples, "tuning_max_samples", false);
    if(max_samples.has_value())
        result.precise.max_samples = result.coarse.max_samples;
    apply(result.sleep_us, sleep_us, "tuning_sleep_us", true);
    return result;
}

// Inner repeat count when timing a candidate, raised for split-k (kernel + prefill).
static std::size_t compute_benchmark_bundle(const module& m)
{
    // Count context-requiring ops (kernel + prefills); skip context-free and @-builtins.
    int n = std::count_if(m.begin(), m.end(), [](const auto& ins) {
        return not migraphx::is_context_free(ins.get_operator()) and
               not starts_with(ins.name(), "@");
    });
    return std::max(1, 4 * n - 2);
}

struct precompile_op
{
    operation op                      = op::identity{};
    std::size_t additional_args       = 1;
    bool ignore_modules               = false;
    std::optional<shape> output_shape = nullopt;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"),
                    f(self.additional_args, "additional_args"),
                    f(self.ignore_modules, "ignore_modules"),
                    f(self.output_shape, "output_shape"));
    }

    std::string name() const { return "gpu::precompile_op"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        // Pop off additional args
        inputs.resize(inputs.size() - additional_args);
        if(output_shape.has_value())
            return output_shape.value();
        if(ignore_modules)
            return op.compute_shape(inputs);
        return op.compute_shape(inputs, mods);
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        return {shapes.size() - 1};
    }
};
MIGRAPHX_REGISTER_OP(precompile_op);

struct dynamic_op_cache
{
    module mod;
    std::vector<shape> input_shapes;
    shape output_shape;
};

struct dynamic_code_object_op
{
    operation pre_op = precompile_op{};

    // This implementation currently caches for each dynamic_code_object_op instance
    // It will be updated to store compiled code objects for all encountered shapes
    //  in a way that can be used by all dynamic_code_object_op instances
    using cache_map_type = std::unordered_map<const dynamic_code_object_op*, dynamic_op_cache>;
    std::shared_ptr<cache_map_type> cache_map = std::make_shared<cache_map_type>();

    dynamic_op_cache& get_cache() const { return (*cache_map)[this]; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.pre_op, "pre_op"));
    }

    std::string name() const { return "gpu::dynamic_code_object_op"; }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        return pre_op.compute_shape(inputs, mods);
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        return {shapes.size() - 1};
    }
    std::unordered_map<std::string, argument> build_param_map(const std::vector<argument>& args,
                                                              const_module_ref mod) const
    {
        auto pnames = mod->get_parameter_names();
        assert(pnames.size() == args.size());
        std::unordered_map<std::string, argument> param_map;
        std::transform(pnames.begin(),
                       pnames.end(),
                       args.begin(),
                       std::inserter(param_map, param_map.end()),
                       [](const auto& name, const auto& arg) { return std::make_pair(name, arg); });
        return param_map;
    }
    argument compute(context& ctx,
                     const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& module_args,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        auto static_args = std::vector<argument>{args.begin(), args.end()};
        auto output_arg  = static_args.back();

        auto& cache = get_cache();
        if(cache.mod.size() > 0 and cache.input_shapes == to_shapes(args))
        {
            static_args[static_args.size() - 1] = output_arg.reshape(cache.output_shape);
            auto* mod                           = &cache.mod;
            auto param_map                      = build_param_map(static_args, mod);
            auto results                        = run(mod, param_map);
            if(results.size() > 1)
                return results;
            return results.front();
        }

        auto out_shape = pre_op.compute_shape(to_shapes(static_args), module_args);
        static_args[static_args.size() - 1] = output_arg.reshape(out_shape);
        // Skip JIT compilation when dynamic shape resolves to 0 elements at runtime
        if(args.front().get_shape().elements() == 0)
            return static_args.back();

        // Rewrite submodule without dynamic shapes to be used as the IR for compilation
        module static_submod;
        auto op_name          = any_cast<precompile_op>(pre_op).op.name();
        auto runtime_mod_name = "runtime_mod:" + op_name;
        if(not module_args.empty())
        {
            auto pnames = module_args.front()->get_parameter_names();
            std::unordered_map<std::string, shape> mod_arg_shapes;
            std::transform(pnames.begin(),
                           pnames.end(),
                           args.begin(),
                           std::inserter(mod_arg_shapes, mod_arg_shapes.end()),
                           [&](const auto& name, const auto& arg) {
                               return std::make_pair(name, arg.get_shape());
                           });
            static_submod = module_args.front()->with_static_shapes(mod_arg_shapes);
            static_submod.set_bypass(true);
            runtime_mod_name = "runtime_mod:" + module_args.front()->name();
        }

        // Create runtime module which will be compiled and cached
        auto runtime_mod = module(runtime_mod_name);
        std::vector<instruction_ref> args_ins;
        std::vector<size_t> idx(static_args.size());
        std::iota(std::begin(idx), std::end(idx), 0);
        std::transform(static_args.begin(),
                       static_args.end(),
                       idx.begin(),
                       std::back_inserter(args_ins),
                       [&](const auto& arg, const auto& i) {
                           return runtime_mod.add_parameter(
                               runtime_mod_name + ":x" + std::to_string(i), arg.get_shape());
                       });
        instruction_ref ins;
        if(not module_args.empty())
        {
            ins = runtime_mod.add_instruction(pre_op, args_ins, {&static_submod});
        }
        else
        {
            ins = runtime_mod.add_instruction(pre_op, args_ins);
        }
        runtime_mod.add_return({ins});

        // Compile ins and replace with a compiled code object op
        operation preop = any_cast<precompile_op>(ins->get_operator()).op;
        auto config     = get_tuning_config(ctx, ins, preop, false);
        value solution  = value{};
        if(config.has_value())
        {
            solution = config->solutions.front();
        }
        auto compiled_op = compile(ctx, ins, preop, solution);
        compiled_op.replace(runtime_mod, ins);
        run_passes(runtime_mod, {dead_code_elimination{}});

        // Finalize the module before execution
        std::vector<migraphx::context> contexts = {migraphx::context(ctx)};
        runtime_mod.finalize(contexts);

        // Update cache
        // TODO: This will be updated to store compiled code objects for all encountered shapes
        cache.mod          = runtime_mod;
        cache.input_shapes = to_shapes(args);
        cache.output_shape = static_args.back().get_shape();

        // Build param_map based on ACTUAL parameters that exist
        module_ref runtime_mod_ref = &runtime_mod;
        auto param_map             = build_param_map(static_args, runtime_mod_ref);

        auto results = run(runtime_mod_ref, param_map);

        if(results.size() > 1)
            return results;
        return results.front();
    }
};
MIGRAPHX_REGISTER_OP(dynamic_code_object_op);

struct compiled_result
{
    compiler_replace replace;
    instruction_ref ins;

    friend std::ostream& operator<<(std::ostream& os, const compiled_result& cr)
    {
        cr.replace.trace(os, cr.ins);
        return os;
    }

    program make_program() const
    {
        program bench_prog;
        auto* mm = bench_prog.get_main_module();

        std::vector<instruction_ref> bench_ins_inputs;
        std::transform(ins->inputs().begin(),
                       ins->inputs().end(),
                       std::back_inserter(bench_ins_inputs),
                       [&](const auto& arg) {
                           return mm->add_parameter(std::to_string(bench_ins_inputs.size()),
                                                    arg->get_shape());
                       });
        auto bench_ins =
            mm->add_instruction(ins->get_operator(), bench_ins_inputs, ins->module_inputs());
        mm->add_return({bench_ins});
        replace.replace(*mm, bench_ins);
        run_passes(*mm,
                   {
                       eliminate_identity{},
                       dead_code_elimination{},
                       memory_coloring{"hip::allocate"},
                   });
        return bench_prog;
    }
};

// Input buffers reused across coarsely timed candidates. Candidates for one problem can allocate
// different workspaces, so the parameter layout is part of the key and not just the fill policy.
struct shared_benchmark_inputs
{
    std::unordered_map<std::string, double> fill_map;
    std::unordered_map<std::string, shape> parameter_shapes;
    std::shared_ptr<parameter_map> params;

    bool matches(const std::unordered_map<std::string, double>& other_fill_map,
                 const std::unordered_map<std::string, shape>& other_shapes) const
    {
        return fill_map == other_fill_map and parameter_shapes == other_shapes;
    }
};

// forward declared since it requires compile_manager
static void replace_inserted_device_ops(context& ctx, module& m);

struct compile_plan
{
    context* ctx;
    operation preop;
    instruction_ref ins;
    module_ref mod;
    optional<tuning_config> config                 = nullopt;
    std::vector<optional<compiled_result>> results = {};
    adaptive_tuning_options tuning                 = {};
    void update_config(bool exhaustive)
    {
        config = get_tuning_config(*ctx, ins, preop, exhaustive);
    }
    template <class Vector>
    void insert_compiles(Vector& compiles, const value& solution, std::size_t i)
    {
        compiles.emplace_back([=] {
            try
            {
                results[i] = compiled_result{compile(*ctx, ins, preop, solution), ins};
            }
            catch(const std::exception& e)
            {
                const auto trace_level = value_of(MIGRAPHX_TRACE_BENCHMARKING{});
                if(trace_level > 0)
                    std::cerr << "Exception in " + preop.name() + ": " + e.what() << std::endl;
                results[i] = nullopt;
            }
            catch(...)
            {
                results[i] = nullopt;
            }
        });
    }

    template <class Vector>
    void add_compiles(Vector& compiles, bool skip_benchmark)
    {
        if(config.has_value())
        {
            const auto& problem = config->problem;
            if(auto sol = ctx->get_problem_cache().get(preop.name(), problem))
            {
                const auto& solution = sol.value();
                // No solution yet until benchmarked so skip for now
                if(solution.is_null())
                    return;
                results.resize(1);
                insert_compiles(compiles, solution, 0);
            }
            else
            {
                const auto& solutions = config->solutions;
                if(solutions.empty())
                    MIGRAPHX_THROW("No solutions provided for " + preop.name() + " with " +
                                   problem_string() + "\n\n" + print_modules());
                const bool dump_mxr =
                    not string_value_of(MIGRAPHX_GPU_DUMP_BENCHMARK_MXR{}).empty();
                if(skip_benchmark or enabled(MIGRAPHX_SKIP_BENCHMARKING{}) or
                   (ctx->is_cross_compile() and not dump_mxr) or solutions.size() == 1)
                {
                    ctx->get_problem_cache().insert(preop.name(), problem, solutions.front());
                    results.resize(1);
                    insert_compiles(compiles, solutions.front(), 0);
                }
                else
                {
                    ctx->get_problem_cache().mark(preop.name(), problem);
                    results.resize(solutions.size());
                    for(auto i : range(solutions.size()))
                    {
                        insert_compiles(compiles, solutions[i], i);
                    }
                }
            }
        }
        else
        {
            results.resize(1);
            insert_compiles(compiles, value{}, 0);
        }
    }
    std::string problem_string() const
    {
        if(config)
            return to_string(config->problem);
        return "<no problem key>";
    }
    std::string print_modules() const
    {
        std::stringstream current_module;
        for(auto* const m : ins->module_inputs())
        {
            current_module << to_string(*m) << "\n";
        }
        std::stringstream submodules;
        for(auto* const m : ins->module_inputs())
        {
            for(auto* const sm : m->get_sub_modules())
            {
                submodules << to_string(*sm) << "\n";
            }
        }
        return (config ? config->detailed_problem_info : "Problem: no config provided") +
               "\n\nModule:\n" + current_module.str() +
               (not submodules.str().empty() ? "\n" + submodules.str() : "") + "Input Shapes:\n" +
               print_input_shapes();
    }
    std::string print_input_shapes() const
    {
        std::stringstream input_shapes;
        for(const auto& i : ins->inputs())
        {
            input_shapes << i->get_shape() << "\n";
        }
        return input_shapes.str();
    }

    const compiled_result& benchmark() const
    {
        const auto trace_level = value_of(MIGRAPHX_TRACE_BENCHMARKING{});
        if(trace_level > 0 and not results.empty())
        {
            std::cout << "Benchmarking " << preop.name() << ": " << results.size() << " configs"
                      << std::endl;
        }
        if(results.empty())
            MIGRAPHX_THROW("No valid tuned compilation for " + preop.name() + " with " +
                           problem_string() + "\n\n" + print_modules());
        if(results.size() == 1)
        {
            if(not results.front().has_value())
                MIGRAPHX_THROW("No valid tuned compilation for " + preop.name() + " with " +
                               problem_string() + "\n\n" + print_modules());
            return *results.front();
        }
        if(not config)
            MIGRAPHX_THROW("Multiple kernels without config for " + preop.name());
        if(trace_level > 1)
            std::cout << "Problem: " << config->problem << std::endl;

        std::vector<std::size_t> valid_indices;
        auto indices = range(results.size());
        std::copy_if(indices.begin(),
                     indices.end(),
                     std::back_inserter(valid_indices),
                     [&](auto i) { return results[i].has_value(); });
        if(valid_indices.empty())
            MIGRAPHX_THROW("No valid tuned compilation for " + preop.name() + " with " +
                           problem_string() + "\n\n" + print_modules());

        if(valid_indices.size() == 1)
        {
            const auto i = valid_indices.front();
            ctx->get_problem_cache().insert(preop.name(), config->problem, config->solutions.at(i));
            return *results[i];
        }

        auto tuning_options = tuning;
        if(tuning_options.top_k >= valid_indices.size())
            tuning_options.top_k = 0;

        // GPU failures tend to be sticky, so one broken candidate usually takes every later one
        // down with it. Keep the first message to report instead of a more general message.
        std::string first_error;
        std::vector<std::size_t> execution_counts(results.size());
        // Coarse candidates share input buffers when their fill policy and parameter layout agree.
        // The buffers are released before precise timing so finalists are measured from fresh
        // state.
        std::vector<shared_benchmark_inputs> shared_inputs;
        auto time_solution = [&](std::size_t i,
                                 adaptive_time_stage stage,
                                 const adaptive_time_options& input_options) -> optional<double> {
            if(not results[i].has_value())
            {
                if(trace_level > 1)
                    std::cout << "No binary for solution: " << config->solutions.at(i) << std::endl;
                return nullopt;
            }

            if(trace_level > 1)
                std::cout << (stage == adaptive_time_stage::coarse ? "Coarsely" : "Precisely")
                          << " benchmarking solution: " << config->solutions.at(i) << std::endl;
            // Held for one timing only so that loaded code objects and input buffers are not
            // retained for every candidate at once.
            optional<prepared_time_program> prepared;
            try
            {
                if(trace_level > 2)
                    std::cout << *results[i] << std::endl;
                // Precise candidates start from fresh programs and parameters, while coarse
                // executions remain charged to each candidate's lifetime execution budget.
                if(stage == adaptive_time_stage::precise)
                    shared_inputs.clear();
                /*
                 * Replacing the instruction in this small program inserts every code object
                 * and prefill required by the candidate, so split-k is timed end to end.
                 */
                auto bench_prog = results[i]->make_program();
                if(trace_level > 2)
                    std::cout << bench_prog << std::endl;
                const auto bundle = compute_benchmark_bundle(*bench_prog.get_main_module());

                // Lifetime budget for this candidate, drawn down by both timing stages.
                const auto candidate_budget = 1 + benchmark_samples * bundle;
                const auto used             = execution_counts[i];
                if(used >= candidate_budget)
                    return nullopt;
                const auto remaining = candidate_budget - used;

                const auto& fill_map  = results[i]->replace.fill_map;
                auto parameter_shapes = bench_prog.get_parameter_shapes();
                auto shared           = shared_inputs.end();
                if(stage == adaptive_time_stage::coarse)
                    shared = std::find_if(
                        shared_inputs.begin(), shared_inputs.end(), [&](const auto& x) {
                            return x.matches(fill_map, parameter_shapes);
                        });
                prepared =
                    prepare_time_program(*ctx,
                                         std::move(bench_prog),
                                         fill_map,
                                         shared == shared_inputs.end() ? nullptr : shared->params);
                if(stage == adaptive_time_stage::coarse and shared == shared_inputs.end())
                    shared_inputs.push_back(
                        {fill_map, std::move(parameter_shapes), prepared->params});
                if(trace_level > 1)
                    std::cout << "Prepared benchmark solution: " << config->solutions.at(i)
                              << std::endl;

                auto options             = input_options;
                options.preferred_bundle = bundle;
                options.max_executions   = std::min(options.max_executions, remaining);
                adaptive_time_budget budget;
                budget.max_executions = remaining;
                if(stage == adaptive_time_stage::coarse)
                {
                    // Coarse timing needs lazy initialization, one estimate, and up to
                    // coarse_samples bundled ranking measurements. Precise timing receives the
                    // lifetime budget left by this call.
                    budget.max_executions =
                        std::min(budget.max_executions, 2 + coarse_samples * bundle);
                }
                auto measured = adaptive_time_program(*prepared, options, budget);
                execution_counts[i] += prepared->executions;
                if(not std::isfinite(measured) or measured <= 0.0)
                    return nullopt;
                if(trace_level > 1)
                    std::cout << measured << "ms" << std::endl;
                return measured;
            }
            catch(const std::exception& e)
            {
                if(prepared.has_value())
                    execution_counts[i] += prepared->executions;
                if(first_error.empty())
                    first_error =
                        "solution " + to_string(config->solutions.at(i)) + ": " + e.what();
                if(trace_level > 0)
                    std::cerr << "Exception benchmarking " << preop.name() << " solution "
                              << config->solutions.at(i) << ": " << e.what() << std::endl;
                return nullopt;
            }
        };

        const auto winner =
            adaptive_time_topk_staged(results.size(), tuning_options, time_solution);
        if(not winner.has_value())
            MIGRAPHX_THROW("No valid tuned benchmark for " + preop.name() + " with " +
                           problem_string() +
                           (first_error.empty() ? "" : "\n\nFirst error: " + first_error) + "\n\n" +
                           print_modules());
        const auto i = *winner;
        ctx->get_problem_cache().insert(preop.name(), config->problem, config->solutions.at(i));
        if(trace_level > 0)
        {
            std::cout << "Fastest solution: " << config->solutions.at(i) << std::endl;
            ctx->get_problem_cache().save();
        }
        if(not results[i].has_value())
            MIGRAPHX_THROW("No valid tuned compilation for " + preop.name() + " with " +
                           problem_string() + "\n\n" + print_modules());
        auto skipped = std::count_if(
            results.begin(), results.end(), [](const auto& cr) { return not cr.has_value(); });
        if(skipped > 0)
            log::info() << "Skipped " << skipped << " configs for " << preop.name();

        return *results[i];
    }

    void replace(module& m) const
    {
        const auto& cr = benchmark();
        cr.replace.replace(m, cr.ins);
    }

    std::size_t save_binaries(const fs::path& mxr_dir) const
    {
        std::size_t saved_files = 0;
        if(not config.has_value())
            return saved_files;
        for(auto i : range(results.size()))
        {
            if(not results[i].has_value())
                continue;
            const auto& solution = config->solutions[i];
            auto bench_prog      = results[i]->make_program();
            auto* mm             = bench_prog.get_main_module();

            replace_inserted_device_ops(*ctx, *mm);

            // Use json encoding for the comment used for benchmarking mxr files.
            value comment_val        = value::object{};
            comment_val["op"]        = preop.name();
            comment_val["problem"]   = config->problem;
            comment_val["solution"]  = solution;
            std::string comment_text = to_json_string(comment_val);

            mm->add_instruction(builtin::comment{comment_text}, {});
            auto problem_hash = std::hash<std::string>{}(to_string(config->problem));
            auto op_filename  = sanitize_filename(preop.name());
            auto mxr_file     = mxr_dir / (op_filename + "_" + std::to_string(i) + "_" +
                                       std::to_string(problem_hash) + ".mxr");
            log::info() << "Saving benchmark binary: " << mxr_file;
            save(bench_prog, mxr_file.string());
            ++saved_files;
        }
        return saved_files;
    }
};

template <class F>
static void par_compile(std::size_t n, F f)
{
    if(n == 0)
        return;
    auto d = value_of(MIGRAPHX_GPU_COMPILE_PARALLEL{});
    if(d == 0)
        d = n;
    par_for(n, n / d, f);
}

struct compile_manager
{
    std::vector<compile_plan> cps;
    bool exhaustive                = false;
    bool skip_benchmark            = false;
    adaptive_tuning_options tuning = {};

    void add_plan(context* ctx, const operation& preop, instruction_ref ins, module_ref mod)
    {
        cps.push_back({ctx, preop, ins, mod, nullopt, {}, tuning});
    }

    void update_configs()
    {
        par_compile(cps.size(), [&](auto i) { cps[i].update_config(exhaustive); });
    }

    void compile(module& m, bool is_root)
    {
        std::vector<std::function<void()>> compiles;
        for(auto& cp : cps)
        {
            cp.add_compiles(compiles, skip_benchmark);
        }
        par_compile(compiles.size(), [&](auto i) { compiles[i](); });

        static const auto mxr_path = string_value_of(MIGRAPHX_GPU_DUMP_BENCHMARK_MXR{});
        bool dump_mxr              = not mxr_path.empty();
        std::size_t dumped_mxr_files = 0;

        if(dump_mxr)
        {
            fs::create_directories(fs::path(mxr_path));
        }

        for(const auto& cp : cps)
        {
            if(cp.results.empty())
                continue;
            if(dump_mxr and cp.results.size() > 1)
            {
                dumped_mxr_files += cp.save_binaries(fs::path(mxr_path));
            }
            else
            {
                cp.replace(m);
            }
        }

        // Exit on the root module so all submodules get processed first.
        if(dump_mxr and is_root)
        {
            if(dumped_mxr_files > 0)
            {
                log::info()
                    << "Benchmark MXR files dumped to " << mxr_path
                    << ". Run the MXR files to create a problem cache, then recompile with the "
                       "cache.";
            }
            else
            {
                log::info() << "MIGRAPHX_GPU_DUMP_BENCHMARK_MXR is set to " << mxr_path
                            << ", but no benchmark files were dumped.";
            }
            std::exit(0);
        }

        // Remove compile_plan already executed
        cps.erase(std::remove_if(cps.begin(),
                                 cps.end(),
                                 [](const auto& cp) { return not cp.results.empty(); }),
                  cps.end());
    }
};

static void replace_inserted_device_ops(context& ctx, module& m)
{
    run_passes(m, {dead_code_elimination{}});
    assert(std::none_of(
        m.begin(), m.end(), [](auto&& ins) { return ins.name() == "gpu::precompile_op"; }));
    run_passes(m, {lower_device_ops{}});
    compile_manager cm;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "gpu::precompile_op")
            continue;
        operation preop = any_cast<precompile_op>(ins->get_operator()).op;
        cm.add_plan(&ctx, preop, ins, &m);
    }
    cm.compile(m, false);
    assert(cm.cps.empty());
}

void compile_ops::apply(module_pass_manager& mpm) const
{
    bool is_root = &mpm.get_module() == mpm.get_root_module();
    auto& m      = mpm.get_module();
    compile_manager cm;
    cm.exhaustive     = exhaustive_tune;
    cm.skip_benchmark = skip_benchmark;
    cm.tuning         = tuning;
    // Find all precompile ops
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "gpu::precompile_op")
            continue;
        operation preop = any_cast<precompile_op>(ins->get_operator()).op;
        cm.add_plan(ctx, preop, ins, &m);
    }
    cm.update_configs();
    cm.compile(m, is_root);
    // Compile already tuned configs
    cm.compile(m, is_root);
    assert(cm.cps.empty());

    replace_inserted_device_ops(*ctx, m);
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

