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
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/time_op.hpp>

#include <mutex>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_COMPILE_PARALLEL);

struct precompile_op
{
    operation op                = op::identity{};
    std::size_t additional_args = 1;
    bool ignore_modules         = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"),
                    f(self.additional_args, "additional_args"),
                    f(self.ignore_modules, "ignore_modules"));
    }

    std::string name() const { return "gpu::precompile_op"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        // Pop off additional args
        inputs.resize(inputs.size() - additional_args);
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

struct problem_cache
{
    bool has(const std::string& name, const value& problem) const
    {
        return contains(cache, create_key(name, problem));
    }
    void insert(const std::string& name, const value& problem, const value& solution)
    {
        assert(not solution.is_null());
        cache[create_key(name, problem)] = solution;
    }
    void mark(const std::string& name, const value& problem)
    {
        cache.insert(std::make_pair(create_key(name, problem), value{}));
    }
    optional<value> get(const std::string& name, const value& problem) const
    {
        auto it = cache.find(create_key(name, problem));
        if(it == cache.end())
            return nullopt;
        return it->second;
    }
    static value create_key(const std::string& name, const value& problem)
    {
        return {{"name", name}, {"problem", problem}};
    }
    std::unordered_map<value, value> cache;
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
            catch(...)
            {
                results[i] = nullopt;
            }
        });
    }

    template <class Vector>
    void add_compiles(Vector& compiles, problem_cache& pc)
    {
        if(config.has_value())
        {
            const auto& problem = config->problem;
            if(auto sol = pc.get(preop.name(), problem))
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
                pc.mark(preop.name(), problem);
                const auto& solutions = config->solutions;
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
    const compiled_result& benchmark(problem_cache& pc) const
    {
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
        std::cout << "Benchmarking " << preop.name() << ": " << results.size() << " configs"
                  << std::endl;
        std::cout << "Problem: " << config->problem << std::endl;
        std::vector<double> times;
        times.reserve(results.size());
        std::transform(
            results.begin(), results.end(), std::back_inserter(times), [&](const auto& cr) {
                if(not cr.has_value())
                    return std::numeric_limits<double>::max();
                return time_op(*ctx, cr->replace.code_object, to_shapes(cr->ins->inputs()), 20)
                    .first;
            });
        auto i = std::distance(times.begin(), std::min_element(times.begin(), times.end()));
        auto j = std::distance(times.begin(), std::max_element(times.begin(), times.end()));
        std::cout << "Fastest solution: " << config->solutions.at(i) << std::endl;
        std::cout << "Slowest solution: " << config->solutions.at(j) << std::endl;
        std::cout << "Fastest time: " << *std::min_element(times.begin(), times.end()) << std::endl;
        std::cout << "Slowest time: " << *std::max_element(times.begin(), times.end()) << std::endl;
        pc.insert(preop.name(), config->problem, config->solutions.at(i));
        if(not results[i].has_value())
            MIGRAPHX_THROW("No valid tuned compilation.");
        return *results[i];
    }
    void replace(module& m, problem_cache& pc) const
    {
        const auto& cr = benchmark(pc);
        cr.replace.replace(m, cr.ins);
    }
};

struct parallel_work
{
    std::size_t start             = 0;
    std::size_t stop              = 0;
    std::shared_ptr<std::mutex> m = std::make_shared<std::mutex>();

    bool empty() const
    {
        std::lock_guard<std::mutex> guard(*m);
        return start >= stop;
    }
    optional<std::size_t> pop()
    {
        std::lock_guard<std::mutex> guard(*m);
        if(start >= stop)
            return nullopt;
        return start++;
    }
};

template <class F>
void par_compile(std::size_t n, F f)
{
    if(n == 0)
        return;
    std::cout << "Compile: " << n << std::endl;
    auto d =
        std::min(n, value_of(MIGRAPHX_GPU_COMPILE_PARALLEL{}, std::thread::hardware_concurrency()));
    std::size_t grainsize = std::ceil(static_cast<double>(n) / d);
    std::vector<parallel_work> pw(d);
    std::size_t work = 0;
    std::generate(pw.begin(), pw.end(), [&] {
        parallel_work p{work, std::min(n, work + grainsize)};
        work += grainsize;
        return p;
    });
    if(work < n)
        MIGRAPHX_THROW("Work missing");
    par_for(d, 1, [&](auto i) {
        while(auto w = pw[i].pop())
            f(*w);
        while(any_of(range(d), [&](auto j) {
            auto k = (j + i + 1) % d;
            if(k == i)
                return false;
            auto w = pw[k].pop();
            if(w.has_value())
                f(*w);
            return w.has_value();
        }))
            ;
    });
}

struct compile_manager
{
    problem_cache pc;
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
            cp.add_compiles(compiles, pc);
        }
        par_compile(compiles.size(), [&](auto i) { compiles[i](); });

        // Replace and/or benchmark
        for(const auto& cp : cps)
        {
            if(cp.results.empty())
                continue;
            cp.replace(m, pc);
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
    // Find all precompile opes
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
