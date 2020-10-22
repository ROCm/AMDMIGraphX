#include "run_verify.hpp"
#include "auto_print.hpp"
#include "verify_program.hpp"
#include <migraphx/env.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_TEST_COMPILE)

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
        return std::move(fut);
    }
    return std::async(std::launch::deferred, std::forward<Function>(f));
}

inline void compile_check(migraphx::program& p, const migraphx::target& t, bool show_trace = false)
{
    auto name   = t.name();
    auto shapes = p.get_output_shapes();
    std::stringstream ss;
    migraphx::compile_options options;
    options.trace = migraphx::tracer{ss};
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

    if(show_trace)
    {
        std::cout << ss.str() << std::endl;
    }
}

target_info run_verify::get_target_info(const std::string& name) const
{
    auto it = info.find(t.name());
    if (it != info.end())
        return it->second;
    else
        return {};
}

void run_verify::validate(const target& t, const migraphx::program& p) const
{
    auto ti = get_target_info(t.name());
    if (ti.validate)
        ti.validate(p);
}

std::vector<migraphx::argument> run_verify::run_ref(migraphx::program p, migraphx::program::parameter_map inputs) const
{
    migraphx::ref::target t{};
    auto_print pp{p, t.name()};
    compile_check(p, t);
    return p.eval(inputs);
}
std::pair<migraphx::program, std::vector<migraphx::argument>> run_verify::run_target(const target& t, migraphx::program p, migraphx::program::parameter_map inputs) const
{
    auto_print pp{p, t.name()};
    auto trace_target = string_value_of(MIGRAPHX_TRACE_TEST_COMPILE{});
    compile_check(p, t, (trace_target == t.name()));
    migraphx::program::parameter_map m;
    for(auto&& input : inputs)
    {
        m[input.first] = t.copy_to(input.second);
    }
    for(auto&& x:p.get_parameter_shapes())
    {
        if (m.count(x.first) == 0)
            m[x.first] = t.allocate(m.second);
    }
    validate(t, p);
    p.eval(m);

    auto tres = p.eval(m);
    std::vector<migraphx::argument> res(tres.size());
    std::transform(tres.begin(), tres.end(), res.begin(), [&](auto& argu) {
        return t.copy_from(argu);
    });

    return res;
}

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

void run_verify::verify(const std::string& name, const program& p) const
{
    using result_future = std::future<std::pair<migraphx::program, std::vector<migraphx::argument>>>;
    std::cout << "[   RUN    ] " << name << std::endl;
    auto_print::set_terminate_handler(name);
    std::vector<std::pair<std::string, result_future>> results;
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::generate_argument(x.second, get_hash(x.first));
    }

    auto gold_f = detach_async([=] { return run_ref(p, m); });
    for(auto name:get_targets())
    {
        if (name == "ref")
            continue;
        target_info ti = get_target_info(name);
        auto t = make_target(name);
        results.emplace_back(name, detach_async([=] { return run_target(t, p, m); }, ti.parallel));
    }

    auto gold = gold_f.get();

    for(auto&& p:results)
    {
        auto name = p.first;
        auto x = p.second.get();
        auto cp = x.first;
        auto result = x.second;

        bool passed = true;
        passed &= (gold.size() == result.size());
        std::size_t num = gold.size();
        for(std::size_t i = 0; ((i < num) and passed); ++i)
        {
            passed &= verify_args(name, gold[i], result[i]);
        }

        if(not passed)
        {
            std::cout << p << std::endl;
            std::cout << "ref:\n" << p << std::endl;
            std::cout << name << ":\n" << cp << std::endl;
            std::cout << std::endl;
        }
    }
    std::set_terminate(nullptr);
    std::cout << "[ COMPLETE ] " << name << std::endl;
}

void run_verify::run(int argc, const char* argv[]) const
{
    std::set<std::string> args(argv + 1, argv + argc);
    const auto& ps = get_programs();
    for(auto&& p:ps)
    {
        if (not args.empty())
        {
            if (args.count(p.name) == 0 and args.count(p.section) == 0)
                continue;
        }
        verify(p.name, p.get_program());
    }
}

void disable_parallel_for(const std::string& name)
{
    info[name].parallel = false;
}
void add_validation_for(const std::string& name, std::function<void(const migraphx::program& p)> v)
{
    info[name] = v;
}
