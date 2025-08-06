/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/compile_bytecode.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/time_op.hpp>
#include <migraphx/gpu/mlir.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_COMPILE_PARALLEL);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_BENCHMARKING);


// For the most part this is just a modified compile_ops file, changed to deal with MLIR bytecode
// important thing here to note is that we do not save the mlir module anywhere, causing us to have to reread
// each bytecode sequence over and over again, this is extremely inefficient. We would need to export mlir.cpp 
// structs/APIs to expose the mlir objects, then we would need to read each bytecode sequence once and when we
// need to compile, affix tuning params, then run, we can just clone the module via: 
//      mlir_module new_module = original_module.clone()
// this might be useful for the general pipeline as well since we won't have to rerun the pipeline from
// start to finish, instead we can start from right before affixing parameters and arch info.

struct bc_compiled_result
{
    mlir_code_object mco;
    instruction_ref ins;

    friend std::ostream& operator<<(std::ostream& os, const bc_compiled_result& cr)
    {
        os << cr.mco.cop.name();
        return os;
    }
};

struct bc_compile_plan
{
    context* ctx;
    operation preop;
    instruction_ref ins;
    module_ref mod;
    optional<tuning_config> config                 = nullopt;
    std::vector<optional<bc_compiled_result>> results = {};
    void update_config(bool exhaustive)
    {
        config = get_tuning_config_mlir(*ctx, ins, exhaustive);
    }
    template <class Vector>
    void insert_compiles(Vector& compiles, const value& solution, std::size_t i)
    {
        compiles.emplace_back([=] {
            try
            {
                /* maybe change what compiled_result is, we dont want to substitute */
                results[i] = bc_compiled_result{compile_mlir(*ctx, ins, any_cast<code_object_op>(preop), solution), ins};
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
        return config->detailed_problem_info + "\n\nModule:\n" + current_module.str() +
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

    const bc_compiled_result& benchmark() const
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
                               cr->mco.cop, bench_ins_inputs, cr->ins->module_inputs());
                           run_passes(*bench_mm, {dead_code_elimination{}});
                           // by default, measure runtime with bundle of 1 benchmark config,
                           // repeat 20 times
                           auto t = time_program(*ctx, bench_prog, std::unordered_map<std::string, double>{}, 1, 20);
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
        ins->replace(cr.mco.cop);
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

struct bc_compile_manager
{
    std::vector<bc_compile_plan> cps;
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

        for(const auto& cp : cps)
        {
            if(cp.results.empty())
                continue;
            cp.replace(m);
        }

        cps.erase(std::remove_if(cps.begin(),
                                 cps.end(),
                                 [](const auto& cp) { return not cp.results.empty(); }),
                  cps.end());
    }
};

void compile_bytecode::apply(module& m) const
{
    bc_compile_manager cm;
    cm.exhaustive = exhaustive_tune;
    // Find all precompile ops
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "gpu::code_object")
            continue;
        
        operation preop = any_cast<code_object_op>(ins->get_operator());

        if(any_cast<code_object_op>(preop).format == code_object_format::binary)
            continue;

        cm.add_plan(ctx, preop, ins, &m);
    }
    cm.update_configs();
    cm.compile(m);
    // Compile already tuned configs
    cm.compile(m);
    assert(cm.cps.empty());
}

} // namespace gpu
} // namespace migraphx
} // namespace MIGRAPHX_INLINE_NS
