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
#include "run_verify.hpp"
#include "auto_print.hpp"
#include "verify_program.hpp"
#include "test.hpp"
#include <migraphx/env.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/verify_args.hpp>
#include <set>

#include <future>
#include <thread>
#include <utility>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_TEST_COMPILE)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_TEST)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DUMP_TEST)

// An improved async, that doesn't block
template <class Function>
std::future<typename std::result_of<Function()>::type> detach_async(Function&& f,
                                                                    bool parallel = true)
{
    if(parallel)
    {
        using result_type = typename std::result_of<Function()>::type;
        std::packaged_task<result_type()> task(std::forward<Function>(f));
        auto fut = task.get_future();
        std::thread(std::move(task)).detach();
        return fut;
    }
    return std::async(std::launch::deferred, std::forward<Function>(f));
}

inline void compile_check(migraphx::program& p, const migraphx::target& t, bool show_trace = false)
{
    auto name   = t.name();
    auto shapes = p.get_output_shapes();
    std::stringstream ss;
    migraphx::compile_options options;
    if(show_trace)
        options.trace = migraphx::tracer{std::cout};
    p.compile(t, options);
    if(shapes.size() != p.get_output_shapes().size())
    {
        std::cout << ss.str() << std::endl;
        throw std::runtime_error("Compiling program with " + name +
                                 " alters its number of outputs");
    }

    auto num = shapes.size();
    for(std::size_t i = 0; i < num; ++i)
    {
        if(p.get_output_shapes()[i].lens() != shapes[i].lens())
        {
            std::cout << ss.str() << std::endl;
            throw std::runtime_error("Compiling program with " + name + " alters its shape");
        }
    }
}

target_info run_verify::get_target_info(const std::string& name) const
{
    auto it = info.find(name);
    if(it != info.end())
        return it->second;
    else
        return {};
}

void run_verify::validate(const migraphx::target& t,
                          const migraphx::program& p,
                          const migraphx::parameter_map& m) const
{
    auto ti = get_target_info(t.name());
    if(ti.validate)
        ti.validate(p, m);
}

std::vector<migraphx::argument> run_verify::run_ref(migraphx::program p,
                                                    migraphx::parameter_map inputs) const
{
    migraphx::ref::target t{};
    auto_print pp{p, t.name()};
    compile_check(p, t);
    return p.eval(std::move(inputs));
}
std::pair<migraphx::program, std::vector<migraphx::argument>> run_verify::run_target(
    const migraphx::target& t, migraphx::program p, const migraphx::parameter_map& inputs) const
{
    auto_print pp{p, t.name()};
    auto trace_target = migraphx::string_value_of(MIGRAPHX_TRACE_TEST_COMPILE{});
    compile_check(p, t, (trace_target == t.name()));
    migraphx::parameter_map m;
    for(auto&& input : inputs)
    {
        m[input.first] = t.copy_to(input.second);
    }
    for(auto&& x : p.get_parameter_shapes())
    {
        if(m.count(x.first) == 0)
        {
            m[x.first] = t.allocate(x.second);
        }
    }
    validate(t, p, m);
    p.eval(m);

    auto tres = p.eval(m);
    std::vector<migraphx::argument> res(tres.size());
    std::transform(
        tres.begin(), tres.end(), res.begin(), [&](auto& argu) { return t.copy_from(argu); });

    return std::make_pair(std::move(p), res);
}

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

void run_verify::verify(const std::string& name, const migraphx::program& p) const
{
    using result_future =
        std::future<std::pair<migraphx::program, std::vector<migraphx::argument>>>;
    auto_print::set_terminate_handler(name);
    if(migraphx::enabled(MIGRAPHX_DUMP_TEST{}))
        migraphx::save(p, name + ".mxr");
    std::vector<std::string> target_names;
    for(const auto& tname : migraphx::get_targets())
    {
        if(tname == "ref")
            continue;

        // if tests disabled, skip running it
        target_info ti = get_target_info(tname);
        if(migraphx::contains(ti.disabled_tests, name))
            continue;

        target_names.push_back(tname);
    }
    if(not target_names.empty())
    {
        std::vector<std::pair<std::string, result_future>> results;
        migraphx::parameter_map m;
        for(auto&& x : p.get_parameter_shapes())
        {
            m[x.first] = migraphx::generate_argument(x.second, get_hash(x.first));
        }

        auto gold_f = detach_async([=] { return run_ref(p, m); });
        for(const auto& tname : target_names)
        {
            target_info ti = get_target_info(tname);
            auto t         = migraphx::make_target(tname);
            results.emplace_back(tname,
                                 detach_async([=] { return run_target(t, p, m); }, ti.parallel));
        }

        assert(gold_f.valid());
        auto gold = gold_f.get();

        for(auto&& pp : results)
        {
            assert(pp.second.valid());
            auto tname  = pp.first;
            auto x      = pp.second.get();
            auto cp     = x.first;
            auto result = x.second;

            bool passed = true;
            passed &= (gold.size() == result.size());
            std::size_t num = gold.size();
            for(std::size_t i = 0; ((i < num) and passed); ++i)
            {
                passed &= migraphx::verify_args(tname, gold[i], result[i]);
            }

            if(not passed or migraphx::enabled(MIGRAPHX_TRACE_TEST{}))
            {
                std::cout << p << std::endl;
                std::cout << "ref:\n" << p << std::endl;
                std::cout << tname << ":\n" << cp << std::endl;
                std::cout << std::endl;
            }
            EXPECT(passed);
        }
    }
    std::set_terminate(nullptr);
}

void run_verify::run(int argc, const char* argv[]) const
{
    std::unordered_map<std::string, std::vector<std::string>> labels;
    for(auto&& p : get_programs())
    {
        labels[p.section].push_back(p.name);
        test::add_test_case(p.name, [=] { verify(p.name, p.get_program()); });
    }
    test::driver d{};
    d.get_case_names = [&](const std::string& name) -> std::vector<std::string> {
        if(labels.count(name) > 0)
            return labels.at(name);
        return {name};
    };
    d.run(argc, argv);
}

void run_verify::disable_parallel_for(const std::string& name) { info[name].parallel = false; }
void run_verify::add_validation_for(const std::string& name, target_info::validation_function v)
{
    info[name].validate = std::move(v);
}

void run_verify::disable_test_for(const std::string& name, const std::vector<std::string>& tests)
{
    auto& disabled_tests = info[name].disabled_tests;
    disabled_tests.insert(disabled_tests.end(), tests.begin(), tests.end());
}
