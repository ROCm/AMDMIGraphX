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
#include <migraphx/pass_manager.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/time_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_COMPILE_PARALLEL);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_BENCHMARKING);

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

    std::shared_ptr<dynamic_op_cache> cache = std::make_shared<dynamic_op_cache>();

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

        if(cache->mod.size() > 0 and cache->input_shapes == to_shapes(args))
        {
            static_args[static_args.size() - 1] = output_arg.reshape(cache->output_shape);
            auto* mod                           = &cache->mod;
            auto param_map                      = build_param_map(static_args, mod);
            auto results                        = run(mod, param_map);
            if(results.size() > 1)
                return results;
            return results.front();
        }

        if(output_arg.get_shape().dynamic())
        {
            auto out_shape = pre_op.compute_shape(to_shapes(static_args), module_args);
            static_args[static_args.size() - 1] = output_arg.reshape(out_shape);
        }

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
        cache->mod          = runtime_mod;
        cache->input_shapes = to_shapes(args);
        cache->output_shape = static_args.back().get_shape();

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
    void add_compiles(Vector& compiles)
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
                ctx->get_problem_cache().mark(preop.name(), problem);
                const auto& solutions = config->solutions;
                if(solutions.empty())
                    MIGRAPHX_THROW("No solutions provided for " + preop.name() + " with " +
                                   problem_string() + "\n\n" + print_modules());
                results.resize(solutions.size());
                for(auto i : range(solutions.size()))
                {
                    auto solution = solutions[i];
                    insert_compiles(compiles, solution, i);
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
                           program bench_prog;
                           auto* bench_mm = bench_prog.get_main_module();
                           std::vector<instruction_ref> bench_ins_inputs;

                           std::transform(cr->ins->inputs().begin(),
                                          cr->ins->inputs().end(),
                                          std::back_inserter(bench_ins_inputs),
                                          [&](const auto& arg) {
                                              return bench_mm->add_parameter(
                                                  std::to_string(bench_ins_inputs.size()),
                                                  arg->get_shape());
                                          });
                           auto bench_ins = bench_mm->add_instruction(
                               cr->ins->get_operator(), bench_ins_inputs, cr->ins->module_inputs());
                           bench_mm->add_return({bench_ins});
                           cr->replace.replace(*bench_mm, bench_ins);
                           // do dead code elimination
                           run_passes(*bench_mm,
                                      {
                                          eliminate_identity{},
                                          dead_code_elimination{},
                                          memory_coloring{"hip::allocate"},
                                      });
                           if(trace_level > 2)
                               std::cout << bench_prog << std::endl;
                           auto t = time_program(*ctx,
                                                 bench_prog,
                                                 cr->replace.fill_map,
                                                 /* bundle */ 10,
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
            std::cout << "Skipped " << skipped << " configs for " << preop.name() << std::endl;

        return *results[i];
    }

    void replace(module& m) const
    {
        const auto& cr = benchmark();
        cr.replace.replace(m, cr.ins);
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

    void compile(module& m)
    {
        std::vector<std::function<void()>> compiles;
        for(auto& cp : cps)
        {
            cp.add_compiles(compiles);
        }
        par_compile(compiles.size(), [&](auto i) { compiles[i](); });

        // Replace and/or benchmark
        for(const auto& cp : cps)
        {
            if(cp.results.empty())
                continue;
            cp.replace(m);
        }

        // Remove compile_plan already executed
        cps.erase(std::remove_if(cps.begin(),
                                 cps.end(),
                                 [](const auto& cp) { return not cp.results.empty(); }),
                  cps.end());
    }
};

void compile_ops::apply(module& m) const
{
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
    cm.compile(m);
    // Compile already tuned configs
    cm.compile(m);
    assert(cm.cps.empty());
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
