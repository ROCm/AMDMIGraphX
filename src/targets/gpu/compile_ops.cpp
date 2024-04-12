/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/algorithm.hpp>
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

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

MIGRAPHX_REGISTER_OP(precompile_op);

struct compiled_result
{
    compiler_replace replace;
    instruction_ref ins;
};

struct compile_plan
{
    context* ctx;
    operation preop;
    instruction_ref ins;
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
                auto solution = sol.value();
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
                                   to_string(problem));
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
    const compiled_result& benchmark() const
    {
        const auto trace_level = value_of(MIGRAPHX_TRACE_BENCHMARKING{});
        if(results.empty())
            MIGRAPHX_THROW("No configs to tune");
        if(results.size() == 1)
        {
            if(not results.front().has_value())
                MIGRAPHX_THROW("No configs to tune");
            return *results.front();
        }
        if(not config)
            MIGRAPHX_THROW("Multiple kernels without config");
        if(trace_level > 0)
            std::cout << "Benchmarking " << preop.name() << ": " << results.size() << " configs"
                      << std::endl;
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
                           // Time all the code objects for a given perf config and calculate total
                           // time e.g. in case of split-K GEMM, it may or may not support fusion.
                           // In that case MLIR compile would return code objects for individual
                           // GEMM and pre/post fusion code objects.
                           auto cobjs = cr->replace.code_objects;
                           double t   = transform_accumulate(
                               cobjs.begin(),
                               cobjs.end(),
                               double{0},
                               std::plus<>{},
                               [&](const operation& op) { return time_op(*ctx, op, 20); });
                           if(trace_level > 1)
                               std::cout << t << "ms" << std::endl;
                           return t;
                       });
        auto i = std::distance(times.begin(), std::min_element(times.begin(), times.end()));
        if(trace_level > 0)
            std::cout << "Fastest solution: " << config->solutions.at(i) << std::endl;
        ctx->get_problem_cache().insert(preop.name(), config->problem, config->solutions.at(i));
        if(not results[i].has_value())
            MIGRAPHX_THROW("No valid tuned compilation for " + preop.name() + " with " +
                           to_string(config->problem));
        auto skipped = std::count_if(
            results.begin(), results.end(), [](const auto& cr) { return not cr.has_value(); });
        if(skipped > 0)
            std::cout << "Skipped " << skipped << " configs for " << preop.name() << std::endl;

        return *results[i];
    }

    instruction_ref
    replace(module& m,
            const std::unordered_map<instruction_ref, instruction_ref>& inputs_rep_map) const
    {
        const auto& cr = benchmark();
        return cr.replace.replace(m, cr.ins, inputs_rep_map);
    }
};

template <class F>
void par_compile(std::size_t n, F f)
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
    std::unordered_map<instruction_ref, instruction_ref> input_rep_map;

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
        for(auto i : range(compiles.size()))
        {
            compiles[i]();
        };

        // Replace and/or benchmark
        for(const auto& cp : cps)
        {
            if(cp.results.empty())
                continue;
            auto rep              = cp.replace(m, input_rep_map);
            input_rep_map[cp.ins] = rep;
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
        cm.add_plan(ctx, preop, ins);
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
