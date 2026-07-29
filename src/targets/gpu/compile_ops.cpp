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
#include <cstdlib>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_COMPILE_PARALLEL);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_BENCHMARKING);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SKIP_BENCHMARKING);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_DUMP_BENCHMARK_MXR);

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

/**
 * Apply a compiler_replace to a standalone module so the replacement can be reused.
 *
 * The instruction is rebuilt in a fresh module whose parameters stand in for its inputs, so the
 * resulting fragment refers to those inputs only by position. Any module arguments are consumed
 * by the replace, leaving a flat module that no longer refers to the program it came from.
 */
static program make_fragment(const compiler_replace& cr, instruction_ref ins)
{
    program frag;
    auto* fm = frag.get_main_module();
    std::vector<instruction_ref> inputs;
    std::transform(ins->inputs().begin(),
                   ins->inputs().end(),
                   std::back_inserter(inputs),
                   [&](const auto& arg) {
                       return fm->add_parameter(compiled_code::input_name(inputs.size()),
                                                arg->get_shape());
                   });
    auto frag_ins = fm->add_instruction(ins->get_operator(), inputs, ins->module_inputs());
    fm->add_return({frag_ins});
    cr.replace(*fm, frag_ins);
    // Only dead code is removed here. eliminate_identity would drop the identity that mlir
    // inserts to order prefills before the kernel, and memory coloring is done later by the
    // module this fragment is spliced into.
    run_passes(*fm, {dead_code_elimination{}});
    return frag;
}

/// Identify the code a compile would produce without producing it. A compiler that cannot
/// describe its output, or that fails trying, gives an empty key, which costs a redundant
/// compile but is always correct.
static std::string compile_key_or_empty(context& ctx,
                                        instruction_ref ins,
                                        const operation& preop,
                                        const value& solution)
{
    try
    {
        return compile_key(ctx, ins, preop, solution);
    }
    catch(...)
    {
        return {};
    }
}

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
        // This runs while the program is being evaluated, so reusing an earlier result matters
        // more here than anywhere else.
        auto key        = compile_key_or_empty(ctx, ins, preop, solution);
        auto& bin_cache = ctx.get_binary_cache();
        compiler_replace compiled_op;
        if(auto cached = bin_cache.get(ctx, key))
        {
            compiled_op.code = *cached;
        }
        else
        {
            compiled_op = compile(ctx, ins, preop, solution);
            if(compiled_op.code.empty())
                compiled_op.code.fragment = make_fragment(compiled_op, ins);
            bin_cache.record_compiled(1);
            binary_cache_entry entry;
            entry.key      = key;
            entry.op_name  = preop.name();
            entry.problem  = config ? config->problem : value{};
            entry.solution = solution;
            entry.code     = compiled_op.code;
            bin_cache.insert(ctx, entry);
        }
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

// forward declared since it requires compile_manager
static void replace_inserted_device_ops(context& ctx, module& m);

/// One compilation a plan is waiting on. Candidates are collected before anything is compiled,
/// so identical ones can be found and compiled only once.
struct compile_candidate
{
    std::size_t plan_index   = 0;
    std::size_t result_index = 0;
    value solution           = {};
    std::string key          = {};
};

struct compile_plan
{
    context* ctx;
    operation preop;
    instruction_ref ins;
    module_ref mod;
    optional<tuning_config> config                 = nullopt;
    std::vector<optional<compiled_result>> results = {};
    void update_config(bool exhaustive)
    {
        config = get_tuning_config(*ctx, ins, preop, exhaustive);
    }

    std::string get_key(const value& solution) const
    {
        return compile_key_or_empty(*ctx, ins, preop, solution);
    }

    /**
     * Compile anyway and check the reused result matches.
     *
     * A key that fails to capture something the compiler depends on does not cause a miss, it
     * hands back the wrong code, so there is nothing to notice at runtime beyond bad numbers.
     * Running with this on turns that into a loud failure.
     */
    void verify_reuse(const value& solution, const compiled_code& reused) const
    {
        auto fresh = compile(*ctx, ins, preop, solution);
        if(fresh.code.empty())
            fresh.code.fragment = make_fragment(fresh, ins);
        if(fresh.code.fragment == reused.fragment)
            return;
        MIGRAPHX_THROW("Binary cache reused a result that does not match a fresh compile of " +
                       preop.name() + ".\nReused:\n" + to_string(reused.fragment) +
                       "\nCompiled:\n" + to_string(fresh.code.fragment));
    }

    optional<compiler_replace> run_compile(const value& solution, const std::string& key) const
    {
        auto& cache = ctx->get_binary_cache();
        if(auto cached = cache.get(*ctx, key))
        {
            compiler_replace cr;
            cr.code = *cached;
            if(cache.verify())
                verify_reuse(solution, *cached);
            return cr;
        }
        try
        {
            auto cr = compile(*ctx, ins, preop, solution);
            // Always replace through the fragment, so the path taken when the result is reused
            // is the only path and cannot drift from the one taken when it is compiled.
            if(cr.code.empty())
                cr.code.fragment = make_fragment(cr, ins);
            cache.record_compiled(1);
            binary_cache_entry entry;
            entry.key      = key;
            entry.op_name  = preop.name();
            entry.problem  = config ? config->problem : value{};
            entry.solution = solution;
            entry.code     = cr.code;
            cache.insert(*ctx, entry);
            return cr;
        }
        catch(const std::exception& e)
        {
            const auto trace_level = value_of(MIGRAPHX_TRACE_BENCHMARKING{});
            if(trace_level > 0)
                std::cerr << "Exception in " + preop.name() + ": " + e.what() << std::endl;
            return nullopt;
        }
        catch(...)
        {
            return nullopt;
        }
    }

    void add_candidates(std::vector<compile_candidate>& candidates, std::size_t plan_index)
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
                candidates.push_back({plan_index, 0, solution});
            }
            else
            {
                const auto& solutions = config->solutions;
                if(solutions.empty())
                    MIGRAPHX_THROW("No solutions provided for " + preop.name() + " with " +
                                   problem_string() + "\n\n" + print_modules());
                const bool dump_mxr =
                    not string_value_of(MIGRAPHX_GPU_DUMP_BENCHMARK_MXR{}).empty();
                if(enabled(MIGRAPHX_SKIP_BENCHMARKING{}) or
                   (ctx->is_cross_compile() and not dump_mxr) or solutions.size() == 1)
                {
                    ctx->get_problem_cache().insert(preop.name(), problem, solutions.front());
                    results.resize(1);
                    candidates.push_back({plan_index, 0, solutions.front()});
                }
                else
                {
                    ctx->get_problem_cache().mark(preop.name(), problem);
                    results.resize(solutions.size());
                    for(auto i : range(solutions.size()))
                    {
                        auto& candidate        = candidates.emplace_back();
                        candidate.plan_index   = plan_index;
                        candidate.result_index = i;
                        candidate.solution     = solutions[i];
                    }
                }
            }
        }
        else
        {
            results.resize(1);
            candidates.push_back({plan_index, 0, value{}});
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
        std::vector<double> times;
        times.reserve(results.size());
        std::transform(results.begin(),
                       results.end(),
                       config->solutions.begin(),
                       std::back_inserter(times),
                       [&](const auto& cr, const auto& solution) {
                           if(trace_level > 1)
                               std::cout << "Benchmarking solution: " << solution << std::endl;
                           if(not cr.has_value())
                           {
                               if(trace_level > 1)
                                   std::cout << "No binary" << std::endl;
                               return std::numeric_limits<double>::max();
                           }
                           if(trace_level > 2)
                               std::cout << *cr << std::endl;
                           /*
                           create a small program with insturction being compiled and call "replace"
                           on that which would insert all the compiled code objects, prefills etc.
                           necessary to run candidate code object
                           */
                           auto bench_prog = cr->make_program();
                           if(trace_level > 2)
                               std::cout << bench_prog << std::endl;
                           auto bundle = compute_benchmark_bundle(*bench_prog.get_main_module());
                           auto t      = time_program(*ctx,
                                                 std::move(bench_prog),
                                                 cr->replace.code.fill_map,
                                                 bundle,
                                                 /* nrun */ 20);
                           if(trace_level > 1)
                               std::cout << t << "ms" << std::endl;
                           return t;
                       });
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        auto i = std::distance(times.begin(), std::min_element(times.begin(), times.end()));
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

    void save_binaries(const fs::path& mxr_dir) const
    {
        if(not config.has_value())
            return;
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
        }
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

/// A compile that one or more plans are waiting on, along with the places its result goes.
struct compile_task
{
    /// The plan whose instruction drives the compile. Any plan in targets would do, since they
    /// all produce the same code.
    std::size_t plan_index                                   = 0;
    value solution                                           = {};
    std::string key                                          = {};
    std::vector<std::pair<std::size_t, std::size_t>> targets = {};
    optional<compiler_replace> result                        = nullopt;
};

struct compile_manager
{
    std::vector<compile_plan> cps;
    bool exhaustive = false;

    template <class... Ts>
    void add_plan(Ts&&... xs)
    {
        cps.push_back({std::forward<Ts>(xs)...});
    }

    void update_configs()
    {
        par_compile(cps.size(), [&](auto i) { cps[i].update_config(exhaustive); });
    }

    /**
     * Group the candidates that would compile to the same code.
     *
     * Since every key is known before any compile starts, each group can be handed to exactly
     * one task and the results shared afterwards, so the compiles never have to coordinate.
     */
    std::vector<compile_task> make_tasks(const std::vector<compile_candidate>& candidates) const
    {
        std::vector<compile_task> tasks;
        std::unordered_map<std::string, std::size_t> task_index;
        for(const auto& c : candidates)
        {
            std::size_t i = tasks.size();
            if(c.key.empty())
            {
                tasks.push_back({c.plan_index, c.solution, c.key});
            }
            else
            {
                auto [it, inserted] = task_index.emplace(c.key, i);
                if(inserted)
                    tasks.push_back({c.plan_index, c.solution, c.key});
                i = it->second;
            }
            tasks[i].targets.emplace_back(c.plan_index, c.result_index);
        }
        return tasks;
    }

    void compile(module& m, bool is_root)
    {
        std::vector<compile_candidate> candidates;
        for(auto i : range(cps.size()))
        {
            cps[i].add_candidates(candidates, i);
        }
        par_compile(candidates.size(), [&](auto i) {
            candidates[i].key = cps[candidates[i].plan_index].get_key(candidates[i].solution);
        });

        auto tasks = make_tasks(candidates);
        if(not cps.empty())
        {
            cps.front().ctx->get_binary_cache().record_reused(candidates.size() - tasks.size());
        }
        par_compile(tasks.size(), [&](auto i) {
            tasks[i].result = cps[tasks[i].plan_index].run_compile(tasks[i].solution, tasks[i].key);
        });

        for(const auto& task : tasks)
        {
            for(auto [pi, ri] : task.targets)
            {
                if(not task.result.has_value())
                {
                    cps[pi].results[ri] = nullopt;
                    continue;
                }
                auto cr = *task.result;
                // The fragment is what gets used from here on. The replace function is bound to
                // the instruction that produced it, so it must not run against another one.
                cr.replace_fn       = nullptr;
                cps[pi].results[ri] = compiled_result{std::move(cr), cps[pi].ins};
            }
        }

        static const auto mxr_path = string_value_of(MIGRAPHX_GPU_DUMP_BENCHMARK_MXR{});
        bool dump_mxr              = not mxr_path.empty();

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
                cp.save_binaries(fs::path(mxr_path));
            }
            else
            {
                cp.replace(m);
            }
        }

        // Only throw on the root module so that submodules (which are processed
        // first by the pass manager and may legitimately have no precompile ops
        // or no multi-solution candidates) don't abort compilation before the
        // root module has had a chance to dump its benchmark MXR files.
        if(dump_mxr and is_root)
        {
            log::info() << "Benchmark MXR files dumped to " << mxr_path
                        << ". Run the MXR files to create a problem cache, then recompile with the "
                           "cache.";
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
    // The cache outlives this pass, since compiling a dynamic shape at evaluation time uses it
    // too, so the settings are recorded on it rather than passed along.
    ctx->get_binary_cache().configure(cache_settings);
    compile_manager cm;
    cm.exhaustive = exhaustive_tune;
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
