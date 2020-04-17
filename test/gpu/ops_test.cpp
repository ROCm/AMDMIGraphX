
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
#include <migraphx/quantization.hpp>

#include <miopen/miopen.h>

#include <future>
#include <thread>

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

struct test_literals : verify_program<test_literals>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input = p.add_literal(
            generate_literal(migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}}));
        auto weights = p.add_literal(
            generate_literal(migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}}));
        auto conv = p.add_instruction(migraphx::op::convolution{}, input, weights);
        p.add_instruction(migraphx::op::relu{}, conv);
        return p;
    }
};

struct test_add : verify_program<test_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x = p.add_parameter("x", s);
        auto y = p.add_parameter("y", s);
        p.add_instruction(migraphx::op::add{}, x, y);
        return p;
    }
};

struct test_add_half : verify_program<test_add_half>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::half_type, {3}};
        auto x = p.add_parameter("x", s);
        auto y = p.add_parameter("y", s);
        p.add_instruction(migraphx::op::add{}, x, y);
        return p;
    }
};

struct test_mul : verify_program<test_mul>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x = p.add_parameter("x", s);
        auto y = p.add_parameter("y", s);
        p.add_instruction(migraphx::op::mul{}, x, y);
        return p;
    }
};

struct test_exp : verify_program<test_exp>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {6}};
        auto x = p.add_instruction(migraphx::op::abs{}, p.add_parameter("x", s));
        p.add_instruction(migraphx::op::exp{}, x);
        return p;
    }
};

struct test_erf : verify_program<test_erf>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 6}};
        auto param = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::erf{}, param);
        return p;
    }
};

struct test_sqrt : verify_program<test_sqrt>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 6}};
        auto param     = p.add_parameter("x", s);
        auto param_abs = p.add_instruction(migraphx::op::abs{}, param);
        p.add_instruction(migraphx::op::sqrt{}, param_abs);
        return p;
    }
};

struct test_sign : verify_program<test_sign>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {2, 3, 4, 6}};
        auto param = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::sign{}, param);
        return p;
    }
};

struct test_log : verify_program<test_log>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {6}};
        auto x = p.add_instruction(migraphx::op::abs{}, p.add_parameter("x", s));
        p.add_instruction(migraphx::op::log{}, x);
        return p;
    }
};

struct test_pow : verify_program<test_pow>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {6}};
        std::vector<float> vec_e(s.elements(), 2.0f);
        auto b = p.add_parameter("x", s);
        auto e = p.add_literal(migraphx::literal(s, vec_e));
        p.add_instruction(migraphx::op::pow{}, b, e);
        return p;
    }
};

struct test_prelu_brcst : verify_program<test_prelu_brcst>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {6}};
        auto x   = p.add_parameter("x", s);
        auto slp = p.add_parameter("slp", s);
        auto r   = p.add_instruction(migraphx::op::prelu{}, x, slp);
        p.add_return({r});

        return p;
    }
};

struct test_sin : verify_program<test_sin>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {10}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::sin{}, x);
        return p;
    }
};

struct test_cos : verify_program<test_cos>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {8}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::cos{}, x);
        return p;
    }
};

struct test_tan : verify_program<test_tan>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::tan{}, x);
        return p;
    }
};

struct test_sinh : verify_program<test_sinh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::sinh{}, x);
        return p;
    }
};

struct test_cosh : verify_program<test_cosh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::cosh{}, x);
        return p;
    }
};

struct test_tanh : verify_program<test_tanh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::tanh{}, x);
        return p;
    }
};

struct test_trans_tanh : verify_program<test_trans_tanh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto tx = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, x);
        auto tanhx = p.add_instruction(migraphx::op::tanh{}, tx);
        auto r     = p.add_instruction(migraphx::op::add{}, tanhx, tanhx);
        p.add_instruction(migraphx::op::contiguous{}, r);

        return p;
    }
};

struct test_trans_tanh1 : verify_program<test_trans_tanh1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto tx = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, x);
        auto tanhx = p.add_instruction(migraphx::op::tanh{}, tx);
        auto r     = p.add_instruction(migraphx::op::add{}, tanhx, tanhx);
        p.add_return({tx, r});

        return p;
    }
};

struct test_slice_sin : verify_program<test_slice_sin>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto l = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
        auto t = p.add_instruction(migraphx::op::slice{{1}, {1}, {2}}, l);
        p.add_instruction(migraphx::op::sin{}, t);

        return p;
    }
};

struct test_asin : verify_program<test_asin>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::asin{}, x);
        return p;
    }
};

struct test_acos : verify_program<test_acos>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::acos{}, x);
        return p;
    }
};

struct test_atan : verify_program<test_atan>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::atan{}, x);
        return p;
    }
};

struct test_asinh : verify_program<test_asinh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::asinh{}, x);
        return p;
    }
};

struct test_acosh : verify_program<test_acosh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {16}};
        auto x       = p.add_parameter("x", s);
        auto min_val = p.add_literal(1.1f);
        auto max_val = p.add_literal(100.0f);
        min_val      = p.add_instruction(migraphx::op::multibroadcast{{16}}, min_val);
        max_val      = p.add_instruction(migraphx::op::multibroadcast{{16}}, max_val);
        auto cx      = p.add_instruction(migraphx::op::clip{}, x, min_val, max_val);
        p.add_instruction(migraphx::op::acosh{}, cx);
        return p;
    }
};

struct test_atanh : verify_program<test_atanh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {16}};
        auto x       = p.add_parameter("x", s);
        auto min_val = p.add_literal(-0.95);
        auto max_val = p.add_literal(0.95);
        min_val      = p.add_instruction(migraphx::op::multibroadcast{{16}}, min_val);
        max_val      = p.add_instruction(migraphx::op::multibroadcast{{16}}, max_val);
        auto cx      = p.add_instruction(migraphx::op::clip{}, x, min_val, max_val);
        p.add_instruction(migraphx::op::atanh{}, cx);
        return p;
    }
};

struct test_scale : verify_program<test_scale>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x     = p.add_parameter("x", s);
        auto y     = p.add_parameter("y", migraphx::shape::float_type);
        auto scale = p.add_instruction(migraphx::op::scalar{s.lens()}, y);
        p.add_instruction(migraphx::op::mul{}, x, scale);
        return p;
    }
};

struct test_slice : verify_program<test_slice>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::int32_type, {2, 2, 4}};
        auto x      = p.add_parameter("x", s);
        auto y      = p.add_parameter("y", {migraphx::shape::int32_type, {2, 2, 2}});
        auto slice0 = p.add_instruction(migraphx::op::slice{{2}, {0}, {2}}, x);
        p.add_instruction(migraphx::op::add{}, y, slice0);

        return p;
    }
};

struct test_triadd : verify_program<test_triadd>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x   = p.add_parameter("x", s);
        auto y   = p.add_parameter("y", s);
        auto z   = p.add_parameter("z", s);
        auto sum = p.add_instruction(migraphx::op::add{}, x, y);
        p.add_instruction(migraphx::op::add{}, sum, z);
        return p;
    }
};

struct test_triadd2 : verify_program<test_triadd2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        migraphx::shape b{migraphx::shape::float_type, {3}};
        auto x   = p.add_parameter("x", s);
        auto y   = p.add_parameter("y", s);
        auto z   = p.add_parameter("z", b);
        auto zb  = p.add_instruction(migraphx::op::broadcast{1, s.lens()}, z);
        auto sum = p.add_instruction(migraphx::op::add{}, x, y);
        p.add_instruction(migraphx::op::add{}, sum, zb);
        return p;
    }
};

struct test_mul_add : verify_program<test_mul_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        migraphx::shape bs{migraphx::shape::float_type, {3}};
        auto x   = p.add_parameter("x", s);
        auto a   = p.add_parameter("a", bs);
        auto b   = p.add_parameter("b", bs);
        auto ab  = p.add_instruction(migraphx::op::broadcast{1, s.lens()}, a);
        auto bb  = p.add_instruction(migraphx::op::broadcast{1, s.lens()}, b);
        auto mul = p.add_instruction(migraphx::op::mul{}, x, ab);
        p.add_instruction(migraphx::op::add{}, mul, bb);
        return p;
    }
};

struct test_add_broadcast : verify_program<test_add_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraphx::shape::float_type, {2, 2, 3}});
        auto y  = p.add_parameter("y", {migraphx::shape::float_type, {2, 2}});
        auto by = p.add_instruction(migraphx::op::broadcast{0, x->get_shape().lens()}, y);
        p.add_instruction(migraphx::op::add{}, x, by);
        return p;
    }
};

struct test_add_broadcast2 : verify_program<test_add_broadcast2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraphx::shape::float_type, {2, 3, 4}});
        auto y  = p.add_parameter("y", {migraphx::shape::float_type, {3}});
        auto by = p.add_instruction(migraphx::op::broadcast{1, x->get_shape().lens()}, y);
        p.add_instruction(migraphx::op::add{}, x, by);
        return p;
    }
};

struct test_add_broadcast3 : verify_program<test_add_broadcast3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraphx::shape::float_type, {2, 4, 5}});
        auto y  = p.add_parameter("y", {migraphx::shape::float_type, {4}});
        auto by = p.add_instruction(migraphx::op::broadcast{1, x->get_shape().lens()}, y);
        p.add_instruction(migraphx::op::add{}, x, by);
        return p;
    }
};

struct test_add_broadcast4 : verify_program<test_add_broadcast4>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraphx::shape::float_type, {2, 3, 5}});
        auto y  = p.add_parameter("y", {migraphx::shape::float_type, {3}});
        auto by = p.add_instruction(migraphx::op::broadcast{1, x->get_shape().lens()}, y);
        p.add_instruction(migraphx::op::add{}, x, by);
        return p;
    }
};

struct test_add_broadcast5 : verify_program<test_add_broadcast5>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraphx::shape::float_type, {2, 4, 8}});
        auto y  = p.add_parameter("y", {migraphx::shape::float_type, {4}});
        auto by = p.add_instruction(migraphx::op::broadcast{1, x->get_shape().lens()}, y);
        p.add_instruction(migraphx::op::add{}, x, by);
        return p;
    }
};

struct test_triadd_broadcast : verify_program<test_triadd_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x   = p.add_parameter("x", {migraphx::shape::float_type, {2, 2, 3}});
        auto y   = p.add_parameter("y", {migraphx::shape::float_type, {2, 2}});
        auto z   = p.add_parameter("z", {migraphx::shape::float_type, {2, 2, 3}});
        auto by  = p.add_instruction(migraphx::op::broadcast{0, x->get_shape().lens()}, y);
        auto sum = p.add_instruction(migraphx::op::add{}, x, by);
        p.add_instruction(migraphx::op::add{}, sum, z);
        return p;
    }
};

struct test_sub : verify_program<test_sub>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x    = p.add_parameter("x", s);
        auto y    = p.add_parameter("y", s);
        auto z    = p.add_parameter("z", s);
        auto diff = p.add_instruction(migraphx::op::sub{}, x, y);
        p.add_instruction(migraphx::op::sub{}, diff, z);
        return p;
    }
};

struct test_sub2 : verify_program<test_sub2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        migraphx::shape b{migraphx::shape::float_type, {3}};
        auto x    = p.add_parameter("x", s);
        auto y    = p.add_parameter("y", s);
        auto z    = p.add_parameter("z", b);
        auto zb   = p.add_instruction(migraphx::op::broadcast{1, s.lens()}, z);
        auto diff = p.add_instruction(migraphx::op::sub{}, x, y);
        p.add_instruction(migraphx::op::sub{}, diff, zb);
        return p;
    }
};

struct test_div : verify_program<test_div>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto x    = p.add_parameter("x", s);
        auto y    = p.add_parameter("y", s);
        auto z    = p.add_parameter("z", s);
        auto diff = p.add_instruction(migraphx::op::div{}, x, y);
        p.add_instruction(migraphx::op::div{}, diff, z);
        return p;
    }
};

struct test_div2 : verify_program<test_div2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        migraphx::shape b{migraphx::shape::float_type, {3}};
        auto x    = p.add_parameter("x", s);
        auto y    = p.add_parameter("y", s);
        auto z    = p.add_parameter("z", b);
        auto zb   = p.add_instruction(migraphx::op::broadcast{1, s.lens()}, z);
        auto diff = p.add_instruction(migraphx::op::div{}, x, y);
        p.add_instruction(migraphx::op::div{}, diff, zb);
        return p;
    }
};

struct test_softmax1 : verify_program<test_softmax1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {5, 3, 3, 4}});
        p.add_instruction(migraphx::op::softmax{0}, x);
        return p;
    }
};

struct test_softmax2 : verify_program<test_softmax2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1000, 1, 1}});
        p.add_instruction(migraphx::op::softmax{}, x);
        return p;
    }
};

template <int Axis, migraphx::shape::type_t T>
struct test_softmax : verify_program<test_softmax<Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{T, {512, 4, 1067, 6}};
        auto param = p.add_parameter("0", s);
        p.add_instruction(migraphx::op::softmax{Axis}, param);

        return p;
    }
};

template struct test_softmax<0, migraphx::shape::float_type>;
template struct test_softmax<2, migraphx::shape::float_type>;
template struct test_softmax<1, migraphx::shape::double_type>;
template struct test_softmax<3, migraphx::shape::double_type>;
template struct test_softmax<0, migraphx::shape::half_type>;
template struct test_softmax<1, migraphx::shape::half_type>;
template struct test_softmax<2, migraphx::shape::half_type>;
template struct test_softmax<3, migraphx::shape::half_type>;

template <class T, int Axis>
struct test_arg_ops : verify_program<test_arg_ops<T, Axis>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 1025}};
        auto param = p.add_parameter("data", s);
        p.add_instruction(T{Axis}, param);

        return p;
    }
};

template struct test_arg_ops<migraphx::op::argmax, 0>;
template struct test_arg_ops<migraphx::op::argmax, 1>;
template struct test_arg_ops<migraphx::op::argmax, 2>;
template struct test_arg_ops<migraphx::op::argmax, 3>;
template struct test_arg_ops<migraphx::op::argmax, -1>;
template struct test_arg_ops<migraphx::op::argmax, -2>;

template struct test_arg_ops<migraphx::op::argmin, 0>;
template struct test_arg_ops<migraphx::op::argmin, 1>;
template struct test_arg_ops<migraphx::op::argmin, 2>;
template struct test_arg_ops<migraphx::op::argmin, 3>;
template struct test_arg_ops<migraphx::op::argmin, -3>;
template struct test_arg_ops<migraphx::op::argmin, -4>;

struct test_conv : verify_program<test_conv>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::convolution{}, input, weights);
        return p;
    }
};

struct test_conv2 : verify_program<test_conv2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 512, 28, 28}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {256, 512, 1, 1}});
        p.add_instruction(migraphx::op::convolution{{0, 0}, {1, 1}, {1, 1}}, input, weights);
        return p;
    }
};

struct test_group_conv : verify_program<test_group_conv>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 4, 16, 16}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 1, 3, 3}});
        migraphx::op::convolution op;
        op.group = 4;
        p.add_instruction(op, input, weights);
        return p;
    }
};

struct test_conv_relu : verify_program<test_conv_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto conv = p.add_instruction(migraphx::op::convolution{}, input, weights);
        p.add_instruction(migraphx::op::relu{}, conv);
        return p;
    }
};

struct test_conv_relu_half : verify_program<test_conv_relu_half>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::half_type, {4, 3, 3, 3}});
        auto conv = p.add_instruction(migraphx::op::convolution{}, input, weights);
        p.add_instruction(migraphx::op::relu{}, conv);
        return p;
    }
};

struct test_conv_bias_clipped_relu : verify_program<test_conv_bias_clipped_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> input_lens{4, 3, 3, 3};
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto l0   = migraphx::literal{migraphx::shape{migraphx::shape::float_type, {4}},
                                    {2.0f, 2.0f, 2.0f, 2.0f}};
        auto bias = p.add_literal(l0);
        auto conv = p.add_instruction(migraphx::op::convolution{}, input, weights);
        auto bcast_add =
            p.add_instruction(migraphx::op::broadcast{1, conv->get_shape().lens()}, bias);
        auto bias_add = p.add_instruction(migraphx::op::add{}, conv, bcast_add);
        auto min_val  = p.add_literal(0.0f);
        auto max_val  = p.add_literal(6.0f);
        min_val       = p.add_instruction(migraphx::op::multibroadcast{input_lens}, min_val);
        max_val       = p.add_instruction(migraphx::op::multibroadcast{input_lens}, max_val);
        p.add_instruction(migraphx::op::clip{}, bias_add, min_val, max_val);
        return p;
    }
};

struct test_conv_add : verify_program<test_conv_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
        auto w = p.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 3, 3}}, 1));
        auto y = p.add_parameter("y", {migraphx::shape::float_type, {1, 8, 4, 4}});
        auto v = p.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 3, 3}}, 2));
        auto conv1 = p.add_instruction(migraphx::op::convolution{}, x, w);
        auto conv2 = p.add_instruction(migraphx::op::convolution{}, y, v);
        auto sum   = p.add_instruction(migraphx::op::add{}, conv1, conv2);
        p.add_instruction(migraphx::op::exp{}, sum);
        return p;
    }
};

struct test_conv_add_1x1_diff_strides : verify_program<test_conv_add_1x1_diff_strides>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", {migraphx::shape::float_type, {1, 8, 2, 2}});
        auto w = p.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 1, 1}}, 1));
        auto y = p.add_parameter("y", {migraphx::shape::float_type, {1, 8, 4, 4}});
        auto v = p.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 1, 1}}, 2));
        auto conv1 = p.add_instruction(migraphx::op::convolution{}, x, w);
        auto conv2 = p.add_instruction(migraphx::op::convolution{{0, 0}, {2, 2}}, y, v);
        auto sum   = p.add_instruction(migraphx::op::add{}, conv1, conv2);
        p.add_instruction(migraphx::op::exp{}, sum);
        return p;
    }
};

struct test_conv_bn_add : verify_program<test_conv_bn_add>
{
    static migraphx::instruction_ref add_bn(migraphx::program& p,
                                            migraphx::instruction_ref x,
                                            std::size_t channels,
                                            std::size_t seed = 1)
    {
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto scale    = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1 + seed)));
        auto bias     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2 + seed)));
        auto mean     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3 + seed)));
        auto variance = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4 + seed)));
        return p.add_instruction(
            migraphx::op::batch_norm_inference{}, x, scale, bias, mean, variance);
    }

    migraphx::program create_program() const
    {
        migraphx::program p;
        std::size_t ichannels = 64;
        std::size_t ochannels = 256;
        auto x     = p.add_parameter("x", {migraphx::shape::float_type, {1, ichannels, 56, 56}});
        auto w     = p.add_literal(migraphx::generate_literal(
            {migraphx::shape::float_type, {ochannels, ichannels, 1, 1}}, 1));
        auto y     = p.add_parameter("y", {migraphx::shape::float_type, {1, ichannels, 56, 56}});
        auto v     = p.add_literal(migraphx::generate_literal(
            {migraphx::shape::float_type, {ochannels, ichannels, 1, 1}}, 2));
        auto relu1 = p.add_instruction(migraphx::op::relu{}, x);
        auto conv1 = p.add_instruction(migraphx::op::convolution{}, relu1, w);
        auto bn1   = add_bn(p, conv1, ochannels, 1);
        auto relu2 = p.add_instruction(migraphx::op::relu{}, y);
        auto conv2 = p.add_instruction(migraphx::op::convolution{}, relu2, v);
        auto bn2   = add_bn(p, conv2, ochannels, 1);
        auto sum   = p.add_instruction(migraphx::op::add{}, bn1, bn2);
        p.add_instruction(migraphx::op::relu{}, sum);
        return p;
    }
};

struct test_add_relu : verify_program<test_add_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x   = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto y   = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto add = p.add_instruction(migraphx::op::add{}, x, y);
        p.add_instruction(migraphx::op::relu{}, add);
        return p;
    }
};

struct test_add_sigmoid : verify_program<test_add_sigmoid>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x   = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto y   = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto add = p.add_instruction(migraphx::op::add{}, x, y);
        p.add_instruction(migraphx::op::sigmoid{}, add);
        return p;
    }
};

struct test_add_tanh : verify_program<test_add_tanh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x   = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto y   = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto add = p.add_instruction(migraphx::op::add{}, x, y);
        p.add_instruction(migraphx::op::tanh{}, add);
        return p;
    }
};

struct test_triadd_relu : verify_program<test_triadd_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x   = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto y   = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto z   = p.add_parameter("z", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto sum = p.add_instruction(migraphx::op::add{}, x, y);
        auto triadd = p.add_instruction(migraphx::op::add{}, sum, z);
        p.add_instruction(migraphx::op::relu{}, triadd);
        return p;
    }
};

struct test_triadd_sigmoid : verify_program<test_triadd_sigmoid>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x   = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto y   = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto z   = p.add_parameter("z", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto sum = p.add_instruction(migraphx::op::add{}, x, y);
        auto triadd = p.add_instruction(migraphx::op::add{}, sum, z);
        p.add_instruction(migraphx::op::sigmoid{}, triadd);
        return p;
    }
};

struct test_triadd_tanh : verify_program<test_triadd_tanh>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x   = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto y   = p.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto z   = p.add_parameter("z", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto sum = p.add_instruction(migraphx::op::add{}, x, y);
        auto triadd = p.add_instruction(migraphx::op::add{}, sum, z);
        p.add_instruction(migraphx::op::tanh{}, triadd);
        return p;
    }
};

struct test_sigmoid : verify_program<test_sigmoid>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::sigmoid{}, x);
        return p;
    }
};

struct test_abs : verify_program<test_abs>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::abs{}, x);
        return p;
    }
};

struct test_trans_abs : verify_program<test_trans_abs>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto tx = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, x);
        auto absx = p.add_instruction(migraphx::op::abs{}, tx);
        auto r    = p.add_instruction(migraphx::op::add{}, absx, absx);
        p.add_instruction(migraphx::op::contiguous{}, r);

        return p;
    }
};

struct test_leaky_relu : verify_program<test_leaky_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::leaky_relu{0.01}, x);
        return p;
    }
};

struct test_elu : verify_program<test_elu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::leaky_relu{1.0}, x);
        return p;
    }
};

struct test_relu_lrn : verify_program<test_relu_lrn>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 2}});
        auto y = p.add_instruction(migraphx::op::relu{}, x);
        p.add_instruction(migraphx::op::lrn{0.0001, 0.75, 1.0, 5}, y);
        return p;
    }
};

struct test_conv_pooling : verify_program<test_conv_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 32, 32}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto conv    = p.add_instruction(migraphx::op::convolution{}, input, weights);
        auto pooling = p.add_instruction(migraphx::op::pooling{"max"}, conv);
        p.add_instruction(migraphx::op::relu{}, pooling);
        return p;
    }
};

struct test_concat_pooling : verify_program<test_concat_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 256, 8, 8}});
        auto transpose = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, input);
        auto concat    = p.add_instruction(migraphx::op::concat{3}, transpose);
        auto concat_t  = p.add_instruction(migraphx::op::transpose{{0, 3, 1, 2}}, concat);

        auto pooling =
            p.add_instruction(migraphx::op::pooling{"average", {0, 0}, {1, 1}, {8, 8}}, concat_t);
        p.add_instruction(migraphx::op::relu{}, pooling);
        return p;
    }
};

struct test_global_avg_pooling : verify_program<test_global_avg_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
        auto op    = migraphx::op::pooling{"average"};
        auto lens  = input->get_shape().lens();
        op.lengths = {lens[2], lens[3]};
        p.add_instruction(op, input);
        return p;
    }
};

struct test_global_max_pooling : verify_program<test_global_max_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
        auto op    = migraphx::op::pooling{"max"};
        auto lens  = input->get_shape().lens();
        op.lengths = {lens[2], lens[3]};
        p.add_instruction(op, input);
        return p;
    }
};

struct test_gemm : verify_program<test_gemm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {4, 5}});
        auto b = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {5, 3}});
        p.add_instruction(migraphx::op::dot{}, a, b);
        return p;
    }
};

struct test_gemm_copy : verify_program<test_gemm_copy>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);
        auto dr = p.add_instruction(migraphx::op::dot{}, pa, pb, pc);
        p.add_instruction(migraphx::op::add{}, dr, dr);

        return p;
    }
};

struct test_gemm_ex : verify_program<test_gemm_ex>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {1, 1, 4, 5}});
        auto b = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 3}});
        p.add_instruction(migraphx::op::dot{}, a, b);
        return p;
    }
};

struct test_gemm_half : verify_program<test_gemm_half>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a = p.add_parameter("a", migraphx::shape{migraphx::shape::half_type, {4, 5}});
        auto b = p.add_parameter("b", migraphx::shape{migraphx::shape::half_type, {5, 3}});
        p.add_instruction(migraphx::op::dot{}, a, b);
        return p;
    }
};

struct test_gemm_ld //: verify_program<test_gemm_ld>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a =
            p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {4, 5}, {10, 1}});
        auto b =
            p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {5, 3}, {20, 1}});
        p.add_instruction(migraphx::op::dot{}, a, b);
        return p;
    }
};

struct test_gemm_transposeb : verify_program<test_gemm_transposeb>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a  = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {4, 5}});
        auto b  = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {3, 5}});
        auto bt = p.add_instruction(migraphx::op::transpose{{1, 0}}, b);
        p.add_instruction(migraphx::op::dot{}, a, bt);
        return p;
    }
};

struct test_gemm_transposeb_ex : verify_program<test_gemm_transposeb_ex>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a  = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {1, 4, 5}});
        auto b  = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {1, 3, 5}});
        auto bt = p.add_instruction(migraphx::op::transpose{{0, 2, 1}}, b);
        p.add_instruction(migraphx::op::dot{}, a, bt);
        return p;
    }
};

struct test_gemm_transposea : verify_program<test_gemm_transposea>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a  = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {5, 4}});
        auto b  = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {5, 3}});
        auto at = p.add_instruction(migraphx::op::transpose{{1, 0}}, a);
        p.add_instruction(migraphx::op::dot{}, at, b);
        return p;
    }
};

struct test_gemm_transposea_ex : verify_program<test_gemm_transposea_ex>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a  = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto b  = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 3}});
        auto at = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, a);
        p.add_instruction(migraphx::op::dot{}, at, b);
        return p;
    }
};

struct test_gemm_transposeab : verify_program<test_gemm_transposeab>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto a  = p.add_parameter("a", migraphx::shape{migraphx::shape::float_type, {5, 4}});
        auto b  = p.add_parameter("b", migraphx::shape{migraphx::shape::float_type, {3, 5}});
        auto at = p.add_instruction(migraphx::op::transpose{{1, 0}}, a);
        auto bt = p.add_instruction(migraphx::op::transpose{{1, 0}}, b);
        p.add_instruction(migraphx::op::dot{}, at, bt);
        return p;
    }
};

struct gemm_multi_dim_2 : verify_program<gemm_multi_dim_2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 4}};
        auto l1 = p.add_parameter("1", m1_shape);
        auto l2 = p.add_parameter("2", m2_shape);

        p.add_instruction(migraphx::op::dot{}, l1, l2);

        return p;
    }
};

struct gemm_2args_mm_1 : verify_program<gemm_2args_mm_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {1, 3, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto l2  = p.add_parameter("2", m2_shape);
        auto bl2 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4}}, l2);

        p.add_instruction(migraphx::op::dot{}, l1, bl2);

        return p;
    }
};

struct gemm_2args_mm_2 : verify_program<gemm_2args_mm_2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {3, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto l2  = p.add_parameter("2", m2_shape);
        auto bl2 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 4}}, l2);

        p.add_instruction(migraphx::op::dot{}, l1, bl2);

        return p;
    }
};

struct gemm_2args_mm_3 : verify_program<gemm_2args_mm_3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {1, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {3, 3, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto bl1 = p.add_instruction(migraphx::op::multibroadcast{{3, 2, 3}}, l1);
        auto l2  = p.add_parameter("2", m2_shape);

        p.add_instruction(migraphx::op::dot{}, bl1, l2);

        return p;
    }
};

struct gemm_2args_mm_4 : verify_program<gemm_2args_mm_4>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {3, 3, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto bl1 = p.add_instruction(migraphx::op::multibroadcast{{3, 2, 3}}, l1);
        auto l2  = p.add_parameter("2", m2_shape);

        p.add_instruction(migraphx::op::dot{}, bl1, l2);

        return p;
    }
};

struct gemm_2args_mm_5 : verify_program<gemm_2args_mm_5>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 1, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto bl1 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 2, 3}}, l1);
        auto l2  = p.add_parameter("2", m2_shape);

        p.add_instruction(migraphx::op::dot{}, bl1, l2);

        return p;
    }
};

struct gemm_2args_mm_6 : verify_program<gemm_2args_mm_6>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 1, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {1, 3, 3, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto bl1 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 2, 3}}, l1);
        auto l2  = p.add_parameter("2", m2_shape);
        auto bl2 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 3, 4}}, l2);

        p.add_instruction(migraphx::op::dot{}, bl1, bl2);

        return p;
    }
};

struct gemm_2args_mm_7 : verify_program<gemm_2args_mm_7>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto bl1 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 2, 3}}, l1);
        auto l2  = p.add_parameter("2", m2_shape);

        p.add_instruction(migraphx::op::dot{}, bl1, l2);

        return p;
    }
};

struct gemm_multi_dim_2_3 : verify_program<gemm_multi_dim_2_3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
        auto l1 = p.add_parameter("1", m1_shape);
        auto l2 = p.add_parameter("2", m2_shape);

        p.add_instruction(migraphx::op::dot{}, l1, l2);

        return p;
    }
};

struct gemm_2args_vv : verify_program<gemm_2args_vv>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {8}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {8}};
        auto l1     = p.add_parameter("1", m1_shape);
        auto ul1    = p.add_instruction(migraphx::op::unsqueeze{{0}}, l1);
        auto l2     = p.add_parameter("2", m2_shape);
        auto ul2    = p.add_instruction(migraphx::op::unsqueeze{{1}}, l2);
        float alpha = 0.23f;

        auto res  = p.add_instruction(migraphx::op::dot{alpha}, ul1, ul2);
        auto sres = p.add_instruction(migraphx::op::squeeze{{0}}, res);
        p.add_instruction(migraphx::op::squeeze{{0}}, sres);

        return p;
    }
};

struct gemm_2args_mv : verify_program<gemm_2args_mv>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {3, 5}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {5}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto l2  = p.add_parameter("2", m2_shape);
        auto ul2 = p.add_instruction(migraphx::op::unsqueeze{{1}}, l2);

        p.add_instruction(migraphx::op::dot{}, l1, ul2);

        return p;
    }
};

struct gemm_2args_bmv : verify_program<gemm_2args_bmv>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 3, 5}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {5}};
        auto l1   = p.add_parameter("1", m1_shape);
        auto l2   = p.add_parameter("2", m2_shape);
        auto ul2  = p.add_instruction(migraphx::op::unsqueeze{{1}}, l2);
        auto bul2 = p.add_instruction(migraphx::op::multibroadcast{{2, 3, 5, 1}}, ul2);

        p.add_instruction(migraphx::op::dot{}, l1, bul2);

        return p;
    }
};

struct gemm_2args_vm : verify_program<gemm_2args_vm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {5}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {5, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto ul1 = p.add_instruction(migraphx::op::unsqueeze{{0}}, l1);
        auto l2  = p.add_parameter("2", m2_shape);

        auto res = p.add_instruction(migraphx::op::dot{}, ul1, l2);
        p.add_instruction(migraphx::op::squeeze{{0}}, res);

        return p;
    }
};

struct gemm_2args_vbm : verify_program<gemm_2args_vbm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {5}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 2, 5, 4}};
        auto l1   = p.add_parameter("1", m1_shape);
        auto ul1  = p.add_instruction(migraphx::op::unsqueeze{{0}}, l1);
        auto bul1 = p.add_instruction(migraphx::op::multibroadcast{{2, 2, 1, 5}}, ul1);

        auto l2 = p.add_parameter("2", m2_shape);

        auto res = p.add_instruction(migraphx::op::dot{}, bul1, l2);
        p.add_instruction(migraphx::op::squeeze{{2}}, res);

        return p;
    }
};

struct gemm_multi_3args : verify_program<gemm_multi_3args>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 3, 3, 2}};
        migraphx::shape m3_shape{migraphx::shape::float_type, {2, 3, 2, 2}};

        auto l1     = p.add_parameter("1", m1_shape);
        auto l2     = p.add_parameter("2", m2_shape);
        auto l3     = p.add_parameter("3", m3_shape);
        float alpha = 0.35;
        float beta  = 0.41;
        p.add_instruction(migraphx::op::dot{alpha, beta}, l1, l2, l3);

        return p;
    }
};

struct gemm_multi_3args_c25 : verify_program<gemm_multi_3args_c25>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {3, 5}};
        migraphx::shape m3_shape{migraphx::shape::float_type, {2, 5}};

        auto l1     = p.add_parameter("1", m1_shape);
        auto l2     = p.add_parameter("2", m2_shape);
        auto l3     = p.add_parameter("3", m3_shape);
        float alpha = 0.35;
        float beta  = 0.41;
        p.add_instruction(migraphx::op::dot{alpha, beta}, l1, l2, l3);

        return p;
    }
};

struct gemm_multi_3args_beta0 : verify_program<gemm_multi_3args_beta0>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {1, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {1, 3, 4}};
        migraphx::shape m3_shape{migraphx::shape::float_type, {1, 2, 4}};
        auto l1 = p.add_parameter("1", m1_shape);
        auto l2 = p.add_parameter("2", m2_shape);
        auto l3 = p.add_parameter("3", m3_shape);

        float alpha = 1.0f;
        float beta  = 0.0f;
        p.add_instruction(migraphx::op::dot{alpha, beta}, l1, l2, l3);

        return p;
    }
};

struct gemm_multi_3args_alpha0 : verify_program<gemm_multi_3args_alpha0>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {1, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {1, 3, 4}};
        migraphx::shape m3_shape{migraphx::shape::float_type, {1, 2, 4}};
        auto l1 = p.add_parameter("1", m1_shape);
        auto l2 = p.add_parameter("2", m2_shape);
        auto l3 = p.add_parameter("3", m3_shape);

        float alpha = 0.0f;
        float beta  = 1.0f;
        p.add_instruction(migraphx::op::dot{alpha, beta}, l1, l2, l3);

        return p;
    }
};

struct gemm_multi_transpose : verify_program<gemm_multi_transpose>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::float_type, {2, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {3, 2, 4}};
        auto l1  = p.add_parameter("1", m1_shape);
        auto l2  = p.add_parameter("2", m2_shape);
        auto tl2 = p.add_instruction(migraphx::op::transpose{{1, 0, 2}}, l2);

        float alpha = 1.0f;
        float beta  = 1.0f;
        p.add_instruction(migraphx::op::dot{alpha, beta}, l1, tl2);

        return p;
    }
};

struct quant_dot_3args_1 : verify_program<quant_dot_3args_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::int8_type, {2, 8}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 7}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};

        auto l1 = p.add_parameter("a", m1_shape);
        auto l2 = p.add_parameter("b", m2_shape);
        auto l3 = p.add_parameter("c", m3_shape);
        p.add_instruction(migraphx::op::quant_dot{}, l1, l2, l3);
        return p;
    }
};

struct quant_dot_3args_2 : verify_program<quant_dot_3args_2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::int8_type, {8, 2}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {8, 7}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};

        auto l1  = p.add_parameter("a", m1_shape);
        auto tl1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l1);
        auto l2  = p.add_parameter("b", m2_shape);
        auto l3  = p.add_parameter("c", m3_shape);
        p.add_instruction(migraphx::op::quant_dot{1, 3}, tl1, l2, l3);
        return p;
    }
};

struct quant_dot_3args_3 : verify_program<quant_dot_3args_3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::int8_type, {2, 8}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {7, 8}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};

        auto l1  = p.add_parameter("a", m1_shape);
        auto l2  = p.add_parameter("b", m2_shape);
        auto tl2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l2);
        auto l3  = p.add_parameter("c", m3_shape);
        p.add_instruction(migraphx::op::quant_dot{2, 3}, l1, tl2, l3);
        return p;
    }
};

struct quant_dot_3args_4 : verify_program<quant_dot_3args_4>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::int8_type, {8, 2}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {7, 8}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {2, 7}};

        auto l1  = p.add_parameter("a", m1_shape);
        auto tl1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l1);
        auto l2  = p.add_parameter("b", m2_shape);
        auto tl2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l2);
        auto l3  = p.add_parameter("c", m3_shape);
        p.add_instruction(migraphx::op::quant_dot{3, 2}, tl1, tl2, l3);
        return p;
    }
};

struct batch_quant_dot_1 : verify_program<batch_quant_dot_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::int8_type, {3, 2, 8, 2}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {3, 2, 7, 8}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {3, 2, 2, 7}};

        auto l1  = p.add_parameter("a", m1_shape);
        auto tl1 = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, l1);
        auto l2  = p.add_parameter("b", m2_shape);
        auto tl2 = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, l2);
        auto l3  = p.add_parameter("c", m3_shape);
        p.add_instruction(migraphx::op::quant_dot{3, 2}, tl1, tl2, l3);
        return p;
    }
};

struct batch_quant_dot_2 : verify_program<batch_quant_dot_2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape m1_shape{migraphx::shape::int8_type, {3, 2, 2, 8}};
        migraphx::shape m2_shape{migraphx::shape::int8_type, {3, 2, 8, 7}};
        migraphx::shape m3_shape{migraphx::shape::int32_type, {3, 2, 2, 7}};

        auto l1 = p.add_parameter("a", m1_shape);
        auto l2 = p.add_parameter("b", m2_shape);
        auto l3 = p.add_parameter("c", m3_shape);
        p.add_instruction(migraphx::op::quant_dot{1, 3}, l1, l2, l3);
        return p;
    }
};

struct test_contiguous : verify_program<test_contiguous>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {4, 4, 4, 3}, {48, 4, 1, 16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::contiguous{}, x);
        EXPECT(p.get_output_shapes().back().standard());
        return p;
    }
};

struct test_contiguous_broadcast : verify_program<test_contiguous_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {1, 2}, {0, 1}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::contiguous{}, x);
        EXPECT(p.get_output_shapes().back().standard());
        return p;
    }
};

struct test_contiguous_broadcast_transpose : verify_program<test_contiguous_broadcast_transpose>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {1, 3072, 768}, {0, 1, 3072}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::contiguous{}, x);
        EXPECT(p.get_output_shapes().back().standard());
        return p;
    }
};

struct test_transpose : verify_program<test_transpose>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {4, 3, 4, 4}};
        auto x                    = p.add_parameter("x", s);
        std::vector<int64_t> perm = {0, 2, 3, 1};
        auto l                    = p.add_instruction(migraphx::op::transpose{perm}, x);
        p.add_instruction(migraphx::op::contiguous{}, l);
        return p;
    }
};

struct test_trans_ret : verify_program<test_trans_ret>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto tx = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, x);
        p.add_return({tx});

        return p;
    }
};

struct test_batchnorm_inference_2 : verify_program<test_batchnorm_inference_2>
{
    const size_t width    = 14;
    const size_t height   = 14;
    const size_t channels = 256;
    const size_t batches  = 1;

    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, height, width}};
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto x        = p.add_parameter("x", s);
        auto scale    = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        p.add_instruction(migraphx::op::batch_norm_inference{}, x, scale, bias, mean, variance);
        return p;
    }
};

struct test_batchnorm_inference : verify_program<test_batchnorm_inference>
{
    const size_t width    = 3;
    const size_t height   = 3;
    const size_t channels = 3;
    const size_t batches  = 4;

    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, height, width}};
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto x        = p.add_parameter("x", s);
        auto scale    = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        p.add_instruction(migraphx::op::batch_norm_inference{}, x, scale, bias, mean, variance);
        return p;
    }
};

struct test_clip : verify_program<test_clip>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto x       = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3}});
        auto min_val = p.add_literal(0.0f);
        auto max_val = p.add_literal(6.0f);
        min_val      = p.add_instruction(migraphx::op::multibroadcast{{3}}, min_val);
        max_val      = p.add_instruction(migraphx::op::multibroadcast{{3}}, max_val);
        p.add_instruction(migraphx::op::clip{}, x, min_val, max_val);
        return p;
    }
};

struct test_conv_bn : verify_program<test_conv_bn>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape xs{migraphx::shape::float_type, {1, 3, 224, 224}};
        migraphx::shape ws{migraphx::shape::float_type, {64, 3, 7, 7}};
        migraphx::shape vars{migraphx::shape::float_type, {64}};
        auto x        = p.add_parameter("x", xs);
        auto w        = p.add_parameter("w", ws);
        auto conv     = p.add_instruction(migraphx::op::convolution{{3, 3}, {2, 2}, {1, 1}}, x, w);
        auto scale    = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        p.add_instruction(migraphx::op::batch_norm_inference{}, conv, scale, bias, mean, variance);
        return p;
    }
};

struct test_conv_bn_relu_pooling : verify_program<test_conv_bn_relu_pooling>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape xs{migraphx::shape::float_type, {1, 3, 224, 224}};
        migraphx::shape ws{migraphx::shape::float_type, {64, 3, 7, 7}};
        migraphx::shape vars{migraphx::shape::float_type, {64}};
        auto x        = p.add_parameter("x", xs);
        auto w        = p.add_parameter("w", ws);
        auto conv     = p.add_instruction(migraphx::op::convolution{{3, 3}, {2, 2}, {1, 1}}, x, w);
        auto scale    = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        auto bn       = p.add_instruction(
            migraphx::op::batch_norm_inference{}, conv, scale, bias, mean, variance);
        auto relu = p.add_instruction(migraphx::op::relu{}, bn);
        p.add_instruction(migraphx::op::pooling{"average", {1, 1}, {2, 2}, {3, 3}}, relu);
        return p;
    }
};

struct quant_conv : verify_program<quant_conv>
{
    migraphx::program create_program()
    {
        migraphx::program p;
        migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
        auto pa = p.add_parameter("a", a_shape);
        migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
        auto pc = p.add_parameter("c", c_shape);
        p.add_instruction(migraphx::op::quant_convolution{}, pa, pc);
        return p;
    }
};

struct quant_conv_default_mode : verify_program<quant_conv_default_mode>
{
    migraphx::program create_program()
    {
        migraphx::program p;
        migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
        auto pa = p.add_parameter("a", a_shape);
        migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
        auto pc = p.add_parameter("c", c_shape);
        p.add_instruction(
            migraphx::op::quant_convolution{{{0, 0}}, {{1, 1}}, {{1, 1}}, migraphx::op::same},
            pa,
            pc);
        return p;
    }
};

struct quant_conv_valid_mode : verify_program<quant_conv_valid_mode>
{
    migraphx::program create_program()
    {
        migraphx::program p;
        migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
        auto pa = p.add_parameter("a", a_shape);
        migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
        auto pc = p.add_parameter("c", c_shape);
        p.add_instruction(
            migraphx::op::quant_convolution{{{0, 0}}, {{1, 1}}, {{1, 1}}, migraphx::op::valid},
            pa,
            pc);
        return p;
    }
};

struct quant_conv_padding : verify_program<quant_conv_padding>
{
    migraphx::program create_program()
    {
        migraphx::program p;
        migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
        auto pa = p.add_parameter("a", a_shape);
        migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
        auto pc = p.add_parameter("c", c_shape);
        p.add_instruction(migraphx::op::quant_convolution{{{1, 1}}, {{1, 1}}}, pa, pc);
        return p;
    }
};

struct quant_conv_padding_stride : verify_program<quant_conv_padding_stride>
{
    migraphx::program create_program()
    {
        migraphx::program p;
        migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
        auto pa = p.add_parameter("a", a_shape);
        migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
        auto pc = p.add_parameter("c", c_shape);
        p.add_instruction(migraphx::op::quant_convolution{{{1, 1}}, {{2, 2}}}, pa, pc);

        return p;
    }
};

struct test_concat_axis_1 : verify_program<test_concat_axis_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        int axis = 1;
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::int32_type, {2, 1}};
        auto l0 = p.add_parameter("x", s0);
        auto l1 = p.add_parameter("y", s1);
        auto l2 = p.add_parameter("z", s2);
        p.add_instruction(migraphx::op::concat{axis}, l0, l1, l2);
        return p;
    }
};

struct test_concat_axis_neg_1 : verify_program<test_concat_axis_neg_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        int axis = -1;
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::int32_type, {2, 1}};
        auto l0 = p.add_parameter("x", s0);
        auto l1 = p.add_parameter("y", s1);
        auto l2 = p.add_parameter("z", s2);
        p.add_instruction(migraphx::op::concat{axis}, l0, l1, l2);
        return p;
    }
};

struct test_concat_axis_0 : verify_program<test_concat_axis_0>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        int axis = 0;
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::int32_type, {1, 2}};
        auto l0 = p.add_parameter("x", s0);
        auto l1 = p.add_parameter("y", s1);
        auto l2 = p.add_parameter("z", s2);
        p.add_instruction(migraphx::op::concat{axis}, l0, l1, l2);
        return p;
    }
};

struct test_concat_transpose : verify_program<test_concat_transpose>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        int axis = 1;
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::int32_type, {2, 4}};
        auto l0  = p.add_parameter("x", s0);
        auto lp1 = p.add_parameter("y", s1);
        auto l1  = p.add_instruction(migraphx::op::transpose{{1, 0}}, lp1);
        auto l2  = p.add_parameter("z", s2);
        p.add_instruction(migraphx::op::concat{axis}, l0, l1, l2);
        return p;
    }
};

struct test_concat_transpose2 : verify_program<test_concat_transpose2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        int axis = 1;
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::int32_type, {5, 2}};
        auto l0  = p.add_parameter("x", s0);
        auto l1  = p.add_parameter("y", s1);
        auto lp2 = p.add_parameter("z", s2);
        auto l2  = p.add_instruction(migraphx::op::transpose{{1, 0}}, lp2);
        p.add_instruction(migraphx::op::concat{axis}, l0, l1, l2);
        return p;
    }
};

struct test_concat_transpose3 : verify_program<test_concat_transpose3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        int axis = 1;
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::int32_type, {5, 2}};
        auto l0  = p.add_parameter("x", s0);
        auto lp1 = p.add_parameter("y", s1);
        auto l1  = p.add_instruction(migraphx::op::transpose{{1, 0}}, lp1);
        auto lp2 = p.add_parameter("z", s2);
        auto l2  = p.add_instruction(migraphx::op::transpose{{1, 0}}, lp2);
        p.add_instruction(migraphx::op::concat{axis}, l0, l1, l2);
        return p;
    }
};

struct test_concat_relu : verify_program<test_concat_relu>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        int axis = 0;
        migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::float_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::float_type, {1, 2}};
        auto l0 = p.add_parameter("x", s0);
        auto l1 = p.add_parameter("y", s1);
        auto l2 = p.add_parameter("z", s2);
        auto r0 = p.add_instruction(migraphx::op::relu{}, l0);
        auto r1 = p.add_instruction(migraphx::op::relu{}, l1);
        auto r2 = p.add_instruction(migraphx::op::relu{}, l2);
        auto c0 = p.add_instruction(migraphx::op::concat{axis}, r0, r1, r2);
        p.add_instruction(migraphx::op::relu{}, c0);
        return p;
    }
};

struct test_pad : verify_program<test_pad>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s0{migraphx::shape::int32_type, {1, 96, 165, 165}};
        std::vector<int64_t> pads0 = {0, 0, 0, 0, 0, 0, 1, 1};
        std::vector<int64_t> pads1 = {0, 0, 0, 0, 1, 1, 1, 1};
        std::vector<int64_t> pads2 = {1, 1, 1, 1, 0, 0, 0, 0};
        std::vector<int64_t> pads3 = {1, 0, 1, 0, 1, 0, 2, 0};
        auto l0                    = p.add_parameter("x", s0);
        p.add_instruction(migraphx::op::pad{pads0}, l0);
        p.add_instruction(migraphx::op::pad{pads1}, l0);
        p.add_instruction(migraphx::op::pad{pads2}, l0);
        p.add_instruction(migraphx::op::pad{pads3}, l0);
        return p;
    }
};

struct test_pad_transposed : verify_program<test_pad_transposed>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::int32_type, {1, 224, 224, 3}};
        auto x = p.add_parameter("x", s);
        auto t = p.add_instruction(migraphx::op::transpose{{0, 3, 1, 2}}, x);
        p.add_instruction(migraphx::op::pad{{0, 0, 2, 2, 0, 0, 3, 3}}, t);
        return p;
    }
};

struct test_pad_int8 : verify_program<test_pad_int8>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<int8_t> data0 = {0, 1, 2, 3};
        migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
        auto l0 = p.add_literal(migraphx::literal{s0, data0});
        migraphx::op::pad op{};
        op.value = std::numeric_limits<int8_t>::lowest();
        op.pads  = {0, 0, 1, 1};
        p.add_instruction(op, l0);
        return p;
    }
};

struct test_pooling_autopad : verify_program<test_pooling_autopad>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s0{migraphx::shape::float_type, {1, 3, 63, 63}};
        auto l0 = p.add_parameter("x", s0);
        migraphx::op::pooling op{"max"};
        op.padding_mode = migraphx::op::padding_mode_t::same;
        op.lengths      = {2, 2};
        op.stride       = {2, 2};
        p.add_instruction(op, l0);
        return p;
    }
};

struct test_gather : verify_program<test_gather>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {2, 2}};
        std::vector<int> indices{1, 2, 2, 1};
        auto a0  = p.add_parameter("data", s);
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = 0;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        return p;
    }
};

struct test_gather_neg_axis : verify_program<test_gather_neg_axis>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {2, 2}};
        std::vector<int> indices{1, 2, 2, 1};
        auto a0  = p.add_parameter("data", s);
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        return p;
    }
};

struct test_gather_neg_indices : verify_program<test_gather_neg_indices>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {2, 2}};
        std::vector<int> indices{-2, -1, -1, -2};
        auto a0  = p.add_parameter("data", s);
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        return p;
    }
};

struct test_gather_scalar_output : verify_program<test_gather_scalar_output>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3}};
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{1};
        auto a0  = p.add_parameter("data", s);
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = 0;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        return p;
    }
};

struct test_gather_scalar_index : verify_program<test_gather_scalar_index>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{1};
        auto a0  = p.add_parameter("data", s);
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        return p;
    }
};

struct test_gather_1d_index : verify_program<test_gather_1d_index>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s_indices{migraphx::shape::int32_type, {1}};
        std::vector<int> indices{1};
        auto a0  = p.add_parameter("data", s);
        auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        p.add_instruction(migraphx::op::gather{axis}, a0, a1);
        return p;
    }
};

void manual_identity()
{
    migraphx::program p;
    std::vector<float> data0 = {0, 1, 2, 3};
    migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
    auto l0 = p.add_literal(migraphx::literal{s0, data0});
    p.add_instruction(migraphx::op::identity{}, l0);
    p.compile(migraphx::gpu::target{});
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second));
    }
    auto result = migraphx::gpu::from_gpu(p.eval(m).back());
    std::cout << result << std::endl;
}

void manual_test_concat_relu()
{
    migraphx::program p;
    int axis                 = 0;
    std::vector<float> data0 = {0, 1, 2, 3};
    std::vector<float> data1 = {4, 5, 6, 7, 8, 9};
    std::vector<float> data2 = {10, 11};
    migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
    migraphx::shape s1{migraphx::shape::float_type, {3, 2}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 2}};
    auto l0 = p.add_literal(migraphx::literal{s0, data0});
    auto l1 = p.add_literal(migraphx::literal{s1, data1});
    auto l2 = p.add_literal(migraphx::literal{s2, data2});
    auto r0 = p.add_instruction(migraphx::op::relu{}, l0);
    auto r1 = p.add_instruction(migraphx::op::relu{}, l1);
    auto r2 = p.add_instruction(migraphx::op::relu{}, l2);
    auto c0 = p.add_instruction(migraphx::op::concat{axis}, r0, r1, r2);
    p.add_instruction(migraphx::op::relu{}, c0);

    p.compile(migraphx::gpu::target{});
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second));
    }
    auto result = migraphx::gpu::from_gpu(p.eval(m).back());
    std::cout << result << std::endl;
}

struct test_conv_bn_relu_pooling2 : verify_program<test_conv_bn_relu_pooling2>
{
    static migraphx::instruction_ref
    add_bn(migraphx::program& p, migraphx::instruction_ref x, std::size_t channels)
    {
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto scale = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1 + channels)));
        auto bias  = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2 + channels)));
        auto mean  = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3 + channels)));
        auto variance =
            p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4 + channels)));
        return p.add_instruction(
            migraphx::op::batch_norm_inference{}, x, scale, bias, mean, variance);
    }
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape xs1{migraphx::shape::float_type, {1, 512, 7, 7}};
        migraphx::shape xs2{migraphx::shape::float_type, {1, 1024, 14, 14}};
        migraphx::shape ws1{migraphx::shape::float_type, {2048, 512, 1, 1}};
        migraphx::shape ws2{migraphx::shape::float_type, {2048, 1024, 1, 1}};
        auto x1    = p.add_parameter("x1", xs1);
        auto w1    = p.add_parameter("w1", ws1);
        auto conv1 = p.add_instruction(migraphx::op::convolution{{0, 0}, {1, 1}, {1, 1}}, x1, w1);
        auto bn1   = add_bn(p, conv1, 2048);
        auto x2    = p.add_parameter("x2", xs2);
        auto w2    = p.add_parameter("w2", ws2);
        auto conv2 = p.add_instruction(migraphx::op::convolution{{0, 0}, {2, 2}, {1, 1}}, x2, w2);
        auto bn2   = add_bn(p, conv2, 2048);
        auto add   = p.add_instruction(migraphx::op::add{}, bn1, bn2);
        auto relu  = p.add_instruction(migraphx::op::relu{}, add);
        p.add_instruction(migraphx::op::pooling{"average", {1, 1}, {2, 2}, {3, 3}}, relu);
        return p;
    }
};

struct test_rnn_forward : verify_program<test_rnn_forward>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output =
            p.add_instruction(migraphx::op::rnn{hidden_size,
                                                {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_rnn_forward10 : verify_program<test_rnn_forward10>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 10;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output =
            p.add_instruction(migraphx::op::rnn{hidden_size,
                                                {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_rnn_two_outputs : verify_program<test_rnn_two_outputs>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 10;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto hs      = p.add_instruction(migraphx::op::rnn{hidden_size,
                                                      {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                                      migraphx::op::rnn_direction::forward,
                                                      clip},
                                    seq,
                                    w,
                                    r,
                                    bias,
                                    und,
                                    ih);
        auto last_hs = p.add_instruction(migraphx::op::rnn_last_output{}, hs);
        p.add_return({hs, last_hs});

        return p;
    }
};

struct test_rnn_reverse : verify_program<test_rnn_reverse>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::reverse,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        return p;
    }
};

struct test_rnn_reverse2 : verify_program<test_rnn_reverse2>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::reverse,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        return p;
    }
};

struct test_rnn_3args : verify_program<test_rnn_3args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};

        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);

        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::reverse,
                                            clip},
                          seq,
                          w,
                          r);

        return p;
    }
};

struct test_rnn_4args : verify_program<test_rnn_4args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 5;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);

        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::reverse,
                                            clip},
                          seq,
                          w,
                          r,
                          bias);

        return p;
    }
};

struct test_rnn_5args : verify_program<test_rnn_5args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 10;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output =
            p.add_instruction(migraphx::op::rnn{hidden_size,
                                                {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_rnn_bidirectional : verify_program<test_rnn_bidirectional>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output =
            p.add_instruction(migraphx::op::rnn{hidden_size,
                                                {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_rnn_bidirectional10 : verify_program<test_rnn_bidirectional10>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 10;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto output =
            p.add_instruction(migraphx::op::rnn{hidden_size,
                                                {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_rnn_bi_3args : verify_program<test_rnn_bi_3args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 10;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto output =
            p.add_instruction(migraphx::op::rnn{hidden_size,
                                                {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_gru_forward_last : verify_program<test_gru_forward_last>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_gru_forward_hs : verify_program<test_gru_forward_hs>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        return p;
    }
};

struct test_gru_forward_3args_und : verify_program<test_gru_forward_3args_und>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip},
                          seq,
                          w,
                          r,
                          und,
                          und,
                          und);

        return p;
    }
};

struct test_gru_forward_3args : verify_program<test_gru_forward_3args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip},
                          seq,
                          w,
                          r);

        return p;
    }
};

struct test_gru_forward_seq1 : verify_program<test_gru_forward_seq1>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip},
                          seq,
                          w,
                          r);

        return p;
    }
};

struct test_gru_forward_default_actv : verify_program<test_gru_forward_default_actv>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(
            migraphx::op::gru{hidden_size, {}, migraphx::op::rnn_direction::forward, clip},
            seq,
            w,
            r);

        return p;
    }
};

struct test_gru_two_outputs : verify_program<test_gru_two_outputs>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto hs  = p.add_instruction(
            migraphx::op::gru{hidden_size, {}, migraphx::op::rnn_direction::forward, clip},
            seq,
            w,
            r);
        auto last_hs = p.add_instruction(migraphx::op::rnn_last_output{}, hs);
        p.add_return({hs, last_hs});

        return p;
    }
};

struct test_gru_forward_default_actv1 : verify_program<test_gru_forward_default_actv1>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(
            migraphx::op::gru{
                hidden_size, {migraphx::op::sigmoid{}}, migraphx::op::rnn_direction::forward, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);

        return p;
    }
};

struct test_gru_reverse_last : verify_program<test_gru_reverse_last>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_gru_reverse_3args : verify_program<test_gru_reverse_3args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::reverse,
                                            clip},
                          seq,
                          w,
                          r);

        return p;
    }
};

struct test_gru_bidirct_last : verify_program<test_gru_bidirct_last>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_gru_bidirct_hs : verify_program<test_gru_bidirct_hs>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        return p;
    }
};

struct test_gru_bidirct_3args_und : verify_program<test_gru_bidirct_3args_und>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip},
                          seq,
                          w,
                          r,
                          und,
                          und,
                          und);

        return p;
    }
};

struct test_gru_bidirct_3args : verify_program<test_gru_bidirct_3args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip},
                          seq,
                          w,
                          r);

        return p;
    }
};

struct test_gru_bidirct_seq1 : verify_program<test_gru_bidirct_seq1>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip},
                          seq,
                          w,
                          r);

        return p;
    }
};

struct test_gru_bidirct_default_actv : verify_program<test_gru_bidirct_default_actv>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(
            migraphx::op::gru{hidden_size, {}, migraphx::op::rnn_direction::bidirectional, clip},
            seq,
            w,
            r);

        return p;
    }
};

struct test_gru_bidirct_default_actv1 : verify_program<test_gru_bidirct_default_actv1>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        return p;
    }
};

struct test_lstm_forward_last : verify_program<test_lstm_forward_last>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto ic   = p.add_parameter("ic", ic_shape);
        auto pph  = p.add_parameter("pph", pph_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output = p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip},
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_lstm_forward_hs : verify_program<test_lstm_forward_hs>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto ic   = p.add_parameter("ic", ic_shape);
        auto pph  = p.add_parameter("pph", pph_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip},
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);

        return p;
    }
};

struct test_lstm_forward_3args_und : verify_program<test_lstm_forward_3args_und>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});
        p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip},
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);

        return p;
    }
};

struct test_lstm_forward_3args : verify_program<test_lstm_forward_3args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip},
            seq,
            w,
            r);

        return p;
    }
};

struct test_lstm_two_outputs : verify_program<test_lstm_two_outputs>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto hs  = p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip},
            seq,
            w,
            r);
        auto last_hs = p.add_instruction(migraphx::op::rnn_last_output{}, hs);
        p.add_return({hs, last_hs});

        return p;
    }
};

struct test_lstm_three_outputs : verify_program<test_lstm_three_outputs>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto hs  = p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip},
            seq,
            w,
            r);
        auto last_hs   = p.add_instruction(migraphx::op::rnn_last_output{}, hs);
        auto last_cell = p.add_instruction(migraphx::op::lstm_last_cell_output{}, hs);
        p.add_return({hs, last_hs, last_cell});

        return p;
    }
};

struct test_lstm_forward_seq1 : verify_program<test_lstm_forward_seq1>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip},
            seq,
            w,
            r);

        return p;
    }
};

struct test_lstm_forward_default_actv : verify_program<test_lstm_forward_default_actv>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(
            migraphx::op::lstm{hidden_size, {}, migraphx::op::rnn_direction::forward, clip},
            seq,
            w,
            r);

        return p;
    }
};

struct test_lstm_forward_default_actv1 : verify_program<test_lstm_forward_default_actv1>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(
            migraphx::op::lstm{
                hidden_size, {migraphx::op::sigmoid{}}, migraphx::op::rnn_direction::forward, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);

        return p;
    }
};

struct test_lstm_reverse_last : verify_program<test_lstm_reverse_last>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto ic   = p.add_parameter("ic", ic_shape);
        auto pph  = p.add_parameter("pph", pph_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output = p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::reverse,
                clip},
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_lstm_reverse_3args : verify_program<test_lstm_reverse_3args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::reverse,
                clip},
            seq,
            w,
            r);

        return p;
    }
};

struct test_lstm_reverse_3args_cell_output : verify_program<test_lstm_reverse_3args_cell_output>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto hs  = p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::reverse,
                clip},
            seq,
            w,
            r);
        p.add_instruction(migraphx::op::lstm_last_cell_output{}, hs);

        return p;
    }
};

struct test_lstm_bidirct_last : verify_program<test_lstm_bidirct_last>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto ic   = p.add_parameter("ic", ic_shape);
        auto pph  = p.add_parameter("pph", pph_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto output = p.add_instruction(
            migraphx::op::lstm{
                hidden_size,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::bidirectional,
                clip},
            seq,
            w,
            r,
            bias,
            und,
            ih,
            ic,
            pph);
        p.add_instruction(migraphx::op::rnn_last_output{}, output);

        return p;
    }
};

struct test_lstm_bidirct_hs : verify_program<test_lstm_bidirct_hs>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::lstm{hidden_size,
                                             {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                             migraphx::op::rnn_direction::bidirectional,
                                             clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        return p;
    }
};

struct test_lstm_bidirct_3args_und : verify_program<test_lstm_bidirct_3args_und>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});
        p.add_instruction(
            migraphx::op::gru{hidden_size,
                              {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                              migraphx::op::rnn_direction::bidirectional,
                              clip},
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);

        return p;
    }
};

struct test_lstm_bidirct_3args : verify_program<test_lstm_bidirct_3args>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(migraphx::op::lstm{hidden_size,
                                             {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                             migraphx::op::rnn_direction::bidirectional,
                                             clip},
                          seq,
                          w,
                          r);

        return p;
    }
};

struct test_lstm_bidirct_seq1 : verify_program<test_lstm_bidirct_seq1>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(migraphx::op::lstm{hidden_size,
                                             {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                             migraphx::op::rnn_direction::bidirectional,
                                             clip},
                          seq,
                          w,
                          r);

        return p;
    }
};

struct test_lstm_bidirct_default_actv : verify_program<test_lstm_bidirct_default_actv>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        p.add_instruction(
            migraphx::op::lstm{hidden_size, {}, migraphx::op::rnn_direction::bidirectional, clip},
            seq,
            w,
            r);

        return p;
    }
};

struct test_lstm_bidirct_default_actv1 : verify_program<test_lstm_bidirct_default_actv1>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::lstm{hidden_size,
                                             {migraphx::op::sigmoid{}},
                                             migraphx::op::rnn_direction::bidirectional,
                                             clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        return p;
    }
};

struct test_lstm_bidirct_default_actv2 : verify_program<test_lstm_bidirct_default_actv2>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::lstm{hidden_size,
                                             {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                             migraphx::op::rnn_direction::bidirectional,
                                             clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        return p;
    }
};

template <int Axis, migraphx::shape::type_t T>
struct test_logsoftmax : verify_program<test_logsoftmax<Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{T, {10, 4, 2080, 6}};
        auto param = p.add_parameter("0", s);
        p.add_instruction(migraphx::op::logsoftmax{Axis}, param);

        return p;
    }
};

template struct test_logsoftmax<0, migraphx::shape::float_type>;
template struct test_logsoftmax<1, migraphx::shape::float_type>;
template struct test_logsoftmax<2, migraphx::shape::float_type>;
template struct test_logsoftmax<3, migraphx::shape::float_type>;
template struct test_logsoftmax<1, migraphx::shape::double_type>;
template struct test_logsoftmax<3, migraphx::shape::double_type>;
template struct test_logsoftmax<1, migraphx::shape::half_type>;
template struct test_logsoftmax<0, migraphx::shape::half_type>;
template struct test_logsoftmax<2, migraphx::shape::half_type>;
template struct test_logsoftmax<3, migraphx::shape::half_type>;

struct test_fp32_fp16_lall : verify_program<test_fp32_fp16_lall>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = p.add_literal(migraphx::literal(s, data));
        auto l2 = p.add_parameter("p2", s);
        p.add_instruction(migraphx::op::add{}, l1, l2);
        migraphx::quantize_fp16(p, {"all"});
        return p;
    };
};

struct test_fp32_fp16_ladd : verify_program<test_fp32_fp16_ladd>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = p.add_literal(migraphx::literal(s, data));
        auto l2 = p.add_parameter("p2", s);
        p.add_instruction(migraphx::op::add{}, l1, l2);
        migraphx::quantize_fp16(p, {"add"});
        return p;
    };
};

struct test_fp32_fp16_add : verify_program<test_fp32_fp16_add>
{
    migraphx::program create_program()
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1   = p.add_parameter("x", s);
        auto p2   = p.add_parameter("y", s);
        auto sum  = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto diff = p.add_instruction(migraphx::op::sub{}, sum, p2);
        p.add_instruction(migraphx::op::add{}, diff, p1);
        migraphx::quantize_fp16(p, {"add"});

        return p;
    };
};

struct test_fp32_fp16_sub : verify_program<test_fp32_fp16_sub>
{
    migraphx::program create_program()
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1   = p.add_parameter("x", s);
        auto p2   = p.add_parameter("y", s);
        auto sum  = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto diff = p.add_instruction(migraphx::op::sub{}, sum, p2);
        p.add_instruction(migraphx::op::add{}, diff, p1);
        migraphx::quantize_fp16(p, {"sub"});

        return p;
    };
};

template <class Op, int Axis, migraphx::shape::type_t T>
struct test_reduce_op_large : verify_program<test_reduce_op_large<Op, Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{T, {3, 1026, 4, 3}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(Op{{1}}, x);
        return p;
    };
};

template struct test_reduce_op_large<migraphx::op::reduce_max, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_mean, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_min, 1, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_prod, 2, migraphx::shape::float_type>;
template struct test_reduce_op_large<migraphx::op::reduce_sum, 1, migraphx::shape::float_type>;

template <class Op, int Axis, migraphx::shape::type_t T>
struct test_reduce_op_small : verify_program<test_reduce_op_small<Op, Axis, T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{T, {3, 4, 8, 8}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(Op{{1}}, x);
        return p;
    };
};
template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max, 2, migraphx::shape::int32_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min, 2, migraphx::shape::int32_type>;

template struct test_reduce_op_small<migraphx::op::reduce_sum, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_mean, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_max, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_min, 2, migraphx::shape::half_type>;
template struct test_reduce_op_small<migraphx::op::reduce_prod, -2, migraphx::shape::half_type>;

struct test_rsqrt : verify_program<test_rsqrt>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        std::vector<size_t> input_lens{1, 3, 16, 16};
        migraphx::shape s{migraphx::shape::float_type, input_lens};
        auto x       = p.add_parameter("x", s);
        auto min_val = p.add_literal(1.0f);
        auto max_val = p.add_literal(std::numeric_limits<float>::max());
        min_val      = p.add_instruction(migraphx::op::multibroadcast{input_lens}, min_val);
        max_val      = p.add_instruction(migraphx::op::multibroadcast{input_lens}, max_val);
        auto l0      = p.add_instruction(migraphx::op::clip{}, x, min_val, max_val);
        p.add_instruction(migraphx::op::rsqrt{}, l0);
        return p;
    };
};

struct test_round : verify_program<test_round>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 6}};
        auto param = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::round{}, param);
        return p;
    };
};

struct test_ceil : verify_program<test_ceil>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::double_type, {2, 3, 4, 6}};
        auto param = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::ceil{}, param);
        return p;
    };
};

struct test_floor : verify_program<test_floor>
{
    migraphx::program create_program() const
    {
        migraphx::program p;

        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 6}};
        auto param = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::floor{}, param);
        return p;
    };
};

struct test_convert : verify_program<test_convert>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {8, 24}};
        migraphx::shape sb{migraphx::shape::float_type, {24, 6}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto ia = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, pa);
        auto ib = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, pb);
        p.add_instruction(migraphx::op::quant_dot{}, ia, ib);

        return p;
    };
};

struct test_recip : verify_program<test_recip>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::double_type, {3}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraphx::op::recip{}, x);
        return p;
    }
};

int main(int argc, const char* argv[]) { test::run(argc, argv); }
