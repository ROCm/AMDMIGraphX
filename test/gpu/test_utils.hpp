#ifndef MIGRAPHX_GUARD_TEST_GPU_TEST_UTILS_HPP
#define MIGRAPHX_GUARD_TEST_GPU_TEST_UTILS_HPP

#include <migraphx/env.hpp>
#include <migraphx/program.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/type_name.hpp>
#include <migraphx/verify_args.hpp>
#include <migraphx/instruction.hpp>

#include <miopen/miopen.h>

#include <future>
#include <thread>
#include <cmath>
#include <numeric>

#include <test.hpp>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_GPU_COMPILE)

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

struct auto_print
{
    static void set_terminate_handler(const std::string& name)
    {
        static std::string pname;
        pname = name;
        std::set_terminate(+[] {
            std::cout << "FAILED: " << pname << std::endl;
            try
            {
                std::rethrow_exception(std::current_exception());
            }
            catch(const std::exception& e)
            {
                std::cout << "    what(): " << e.what() << std::endl;
            }
            std::cout << std::endl;
            for(auto&& handle : auto_print::handlers)
                handle();
        });
    }
    static std::array<std::function<void()>, 2> handlers;
    int index;
    template <class T>
    auto_print(T& x, int i) : index(i)
    {
        handlers[index] = [&x] { std::cout << x << std::endl; };
    }

    ~auto_print()
    {
        handlers[index] = [] {};
    }
};
std::array<std::function<void()>, 2> auto_print::handlers = {};

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

void compile_check(migraphx::program& p, const migraphx::target& t, bool show_trace = false)
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

template <class V>
std::vector<migraphx::argument> run_cpu(migraphx::program& p)
{
    V v;
    p = v.create_program();
    auto_print pp{p, 0};
    compile_check(p, migraphx::cpu::target{});
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::generate_argument(x.second, get_hash(x.first));
    }
    return p.eval(m);
}

template <class V>
std::vector<migraphx::argument> run_gpu(migraphx::program& p)
{
    V v;
    p = v.create_program();
    auto_print pp{p, 1};
    compile_check(p, migraphx::gpu::target{}, migraphx::enabled(MIGRAPHX_TRACE_GPU_COMPILE{}));
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] =
            migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
    }
    // Program should have an output parameter
    EXPECT(std::any_of(
        m.begin(), m.end(), [](auto& x) { return migraphx::contains(x.first, "output"); }));

    // Ensure the program doesn't modify the context in a dry run
    auto ctx = p.get_context();
    assert(&ctx != &p.get_context());
    EXPECT(is_shared(ctx, p.get_context()));
    p.dry_run(m);
    EXPECT(is_shared(ctx, p.get_context()));
    p.eval(m);

    auto gpu_res = p.eval(m);
    std::vector<migraphx::argument> res(gpu_res.size());
    std::transform(gpu_res.begin(), gpu_res.end(), res.begin(), [&](auto& argu) {
        return migraphx::gpu::from_gpu(argu);
    });

    return res;
}

template <class V>
void run_verify_program()
{
    auto_print::set_terminate_handler(migraphx::get_type_name<V>());
    // std::cout << migraphx::get_type_name<V>() << std::endl;
    migraphx::program cpu_prog;
    migraphx::program gpu_prog;
    auto cpu_arg_f = detach_async([&] { return run_cpu<V>(cpu_prog); });
    auto gpu_arg   = run_gpu<V>(gpu_prog);
    auto cpu_arg   = cpu_arg_f.get();

    bool passed = true;
    passed &= (cpu_arg.size() == gpu_arg.size());
    std::size_t num = cpu_arg.size();
    for(std::size_t i = 0; ((i < num) and passed); ++i)
    {
        passed &= verify_args(migraphx::get_type_name<V>(), cpu_arg[i], gpu_arg[i]);
    }

    if(not passed)
    {
        V v;
        auto p = v.create_program();
        std::cout << p << std::endl;
        std::cout << "cpu:\n" << cpu_prog << std::endl;
        std::cout << "gpu:\n" << gpu_prog << std::endl;
        std::cout << std::endl;
    }
    std::set_terminate(nullptr);
}

template <class T>
int auto_register_verify_program()
{
    test::add_test_case(migraphx::get_type_name<T>(), [] { run_verify_program<T>(); });
    return 0;
}

template <class T>
struct verify_program
{
    static int static_register;
    // This typedef ensures that the static member will be instantiated if
    // the class itself is instantiated
    using static_register_type =
        std::integral_constant<decltype(&static_register), &static_register>;
};

template <class T>
int verify_program<T>::static_register = auto_register_verify_program<T>(); // NOLINT

#endif
