
#include <migraph/program.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
#include <migraph/cpu/cpu_target.hpp>
#include <migraph/gpu/target.hpp>
#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/type_name.hpp>
#include <migraph/verify_args.hpp>

#include <miopen/miopen.h>

#include <future>
#include <thread>

#include "test.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

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
    else
    {
        return std::async(std::launch::deferred, std::forward<Function>(f));
    }
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

void compile_check(migraph::program& p, const migraph::target& t)
{
    auto name = t.name();
    auto s    = p.get_shape();
    std::stringstream ss;
    p.compile(t, migraph::tracer{ss});
    if(p.get_shape() != s)
    {
        std::cout << ss.str() << std::endl;
        throw std::runtime_error("Compiling program with " + name + " alters its shape");
    }
}

template <class V>
migraph::argument run_cpu()
{
    V v;
    auto p = v.create_program();
    auto_print pp{p, 0};
    compile_check(p, migraph::cpu::cpu_target{});
    migraph::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraph::generate_argument(x.second, get_hash(x.first));
    }
    return p.eval(m);
}

template <class V>
migraph::argument run_gpu()
{
    V v;
    auto p = v.create_program();
    auto_print pp{p, 1};
    compile_check(p, migraph::gpu::target{});
    migraph::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraph::gpu::to_gpu(migraph::generate_argument(x.second, get_hash(x.first)));
    }
    EXPECT(bool{m.find("output") != m.end()});
    return migraph::gpu::from_gpu(p.eval(m));
}

template <class V>
void verify_program()
{
    auto_print::set_terminate_handler(migraph::get_type_name<V>());
    auto cpu_arg_f = detach_async([] { return run_cpu<V>(); });
    auto gpu_arg   = run_gpu<V>();
    verify_args(migraph::get_type_name<V>(), cpu_arg_f.get(), gpu_arg);
    std::set_terminate(nullptr);
}

struct test_literals
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto input = p.add_literal(
            generate_literal(migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}}));
        auto weights = p.add_literal(
            generate_literal(migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}}));
        auto conv = p.add_instruction(migraph::convolution{}, input, weights);
        p.add_instruction(migraph::activation{"relu"}, conv);
        return p;
    }
};

struct test_add
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {3}};
        auto x = p.add_parameter("x", s);
        auto y = p.add_parameter("y", s);
        p.add_instruction(migraph::add{}, x, y);
        return p;
    }
};

struct test_add_broadcast
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraph::shape::float_type, {2, 2, 3}});
        auto y  = p.add_parameter("y", {migraph::shape::float_type, {2, 2}});
        auto by = p.add_instruction(migraph::broadcast{0}, x, y);
        p.add_instruction(migraph::add{}, x, by);
        return p;
    }
};

struct test_add_broadcast2
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraph::shape::float_type, {2, 3, 4}});
        auto y  = p.add_parameter("y", {migraph::shape::float_type, {3}});
        auto by = p.add_instruction(migraph::broadcast{1}, x, y);
        p.add_instruction(migraph::add{}, x, by);
        return p;
    }
};

struct test_add_broadcast3
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraph::shape::float_type, {2, 4, 5}});
        auto y  = p.add_parameter("y", {migraph::shape::float_type, {4}});
        auto by = p.add_instruction(migraph::broadcast{1}, x, y);
        p.add_instruction(migraph::add{}, x, by);
        return p;
    }
};

struct test_add_broadcast4
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraph::shape::float_type, {2, 3, 5}});
        auto y  = p.add_parameter("y", {migraph::shape::float_type, {3}});
        auto by = p.add_instruction(migraph::broadcast{1}, x, y);
        p.add_instruction(migraph::add{}, x, by);
        return p;
    }
};

struct test_add_broadcast5
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraph::shape::float_type, {2, 4, 8}});
        auto y  = p.add_parameter("y", {migraph::shape::float_type, {4}});
        auto by = p.add_instruction(migraph::broadcast{1}, x, y);
        p.add_instruction(migraph::add{}, x, by);
        return p;
    }
};

struct test_softmax
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto x = p.add_parameter("x", migraph::shape{migraph::shape::float_type, {5, 3, 4, 2}});
        p.add_instruction(migraph::softmax{}, x);
        return p;
    }
};

struct test_softmax2
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto x = p.add_parameter("x", migraph::shape{migraph::shape::float_type, {1, 1000}});
        p.add_instruction(migraph::softmax{}, x);
        return p;
    }
};

struct test_conv
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto input = p.add_parameter("x", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraph::convolution{}, input, weights);
        return p;
    }
};

struct test_conv_relu
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto input = p.add_parameter("x", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        auto conv = p.add_instruction(migraph::convolution{}, input, weights);
        p.add_instruction(migraph::activation{"relu"}, conv);
        return p;
    }
};

struct test_add_relu
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto x   = p.add_parameter("x", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        auto y   = p.add_parameter("y", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        auto add = p.add_instruction(migraph::add{}, x, y);
        p.add_instruction(migraph::activation{"relu"}, add);
        return p;
    }
};

struct test_conv_pooling
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto input =
            p.add_parameter("x", migraph::shape{migraph::shape::float_type, {4, 3, 32, 32}});
        auto weights =
            p.add_parameter("w", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        auto conv    = p.add_instruction(migraph::convolution{}, input, weights);
        auto pooling = p.add_instruction(migraph::pooling{"max"}, conv);
        p.add_instruction(migraph::activation{"relu"}, pooling);
        return p;
    }
};

struct test_gemm
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto a = p.add_parameter("a", migraph::shape{migraph::shape::float_type, {4, 5}});
        auto b = p.add_parameter("b", migraph::shape{migraph::shape::float_type, {5, 3}});
        p.add_instruction(migraph::gemm{}, a, b);
        return p;
    }
};

struct test_gemm_ld
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto a = p.add_parameter("a", migraph::shape{migraph::shape::float_type, {4, 5}, {10, 1}});
        auto b = p.add_parameter("b", migraph::shape{migraph::shape::float_type, {5, 3}, {20, 1}});
        p.add_instruction(migraph::gemm{}, a, b);
        return p;
    }
};

struct test_gemm_transposeb
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto a  = p.add_parameter("a", migraph::shape{migraph::shape::float_type, {4, 5}});
        auto b  = p.add_parameter("b", migraph::shape{migraph::shape::float_type, {3, 5}});
        auto bt = p.add_instruction(migraph::transpose{{1, 0}}, b);
        p.add_instruction(migraph::gemm{}, a, bt);
        return p;
    }
};

struct test_gemm_transposea
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto a  = p.add_parameter("a", migraph::shape{migraph::shape::float_type, {5, 4}});
        auto b  = p.add_parameter("b", migraph::shape{migraph::shape::float_type, {5, 3}});
        auto at = p.add_instruction(migraph::transpose{{1, 0}}, a);
        p.add_instruction(migraph::gemm{}, at, b);
        return p;
    }
};

struct test_gemm_transposeab
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto a  = p.add_parameter("a", migraph::shape{migraph::shape::float_type, {5, 4}});
        auto b  = p.add_parameter("b", migraph::shape{migraph::shape::float_type, {3, 5}});
        auto at = p.add_instruction(migraph::transpose{{1, 0}}, a);
        auto bt = p.add_instruction(migraph::transpose{{1, 0}}, b);
        p.add_instruction(migraph::gemm{}, at, bt);
        return p;
    }
};

struct test_contiguous
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {4, 4, 4, 3}, {48, 4, 1, 16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraph::contiguous{}, x);
        EXPECT(p.get_shape().standard());
        return p;
    }
};

struct test_transpose
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {4, 3, 4, 4}};
        auto x                    = p.add_parameter("x", s);
        std::vector<int64_t> perm = {0, 2, 3, 1};
        auto l                    = p.add_instruction(migraph::transpose{perm}, x);
        p.add_instruction(migraph::contiguous{}, l);
        return p;
    }
};

struct test_batchnorm_inference_2
{
    const size_t width    = 14;
    const size_t height   = 14;
    const size_t channels = 256;
    const size_t batches  = 1;

    migraph::program create_program() const
    {
        migraph::program p;

        migraph::shape s{migraph::shape::float_type, {batches, channels, height, width}};
        migraph::shape vars{migraph::shape::float_type, {channels}};
        auto x        = p.add_parameter("x", s);
        auto scale    = p.add_literal(migraph::abs(migraph::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraph::abs(migraph::generate_literal(vars, 4)));
        p.add_instruction(migraph::batch_norm_inference{}, x, scale, bias, mean, variance);
        return p;
    }
};

struct test_batchnorm_inference
{
    const size_t width    = 3;
    const size_t height   = 3;
    const size_t channels = 3;
    const size_t batches  = 4;

    migraph::program create_program() const
    {
        migraph::program p;

        migraph::shape s{migraph::shape::float_type, {batches, channels, height, width}};
        migraph::shape vars{migraph::shape::float_type, {channels}};
        auto x        = p.add_parameter("x", s);
        auto scale    = p.add_literal(migraph::abs(migraph::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraph::abs(migraph::generate_literal(vars, 4)));
        p.add_instruction(migraph::batch_norm_inference{}, x, scale, bias, mean, variance);
        return p;
    }
};

struct test_conv_bn
{
    migraph::program create_program() const
    {
        migraph::program p;

        migraph::shape xs{migraph::shape::float_type, {1, 3, 224, 224}};
        migraph::shape ws{migraph::shape::float_type, {64, 3, 7, 7}};
        migraph::shape vars{migraph::shape::float_type, {64}};
        auto x        = p.add_parameter("x", xs);
        auto w        = p.add_parameter("w", ws);
        auto conv     = p.add_instruction(migraph::convolution{{3, 3}, {2, 2}, {1, 1}}, x, w);
        auto scale    = p.add_literal(migraph::abs(migraph::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraph::abs(migraph::generate_literal(vars, 4)));
        p.add_instruction(migraph::batch_norm_inference{}, conv, scale, bias, mean, variance);
        return p;
    }
};

struct test_conv_bn_relu_pooling
{
    migraph::program create_program() const
    {
        migraph::program p;

        migraph::shape xs{migraph::shape::float_type, {1, 3, 224, 224}};
        migraph::shape ws{migraph::shape::float_type, {64, 3, 7, 7}};
        migraph::shape vars{migraph::shape::float_type, {64}};
        auto x        = p.add_parameter("x", xs);
        auto w        = p.add_parameter("w", ws);
        auto conv     = p.add_instruction(migraph::convolution{{3, 3}, {2, 2}, {1, 1}}, x, w);
        auto scale    = p.add_literal(migraph::abs(migraph::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraph::abs(migraph::generate_literal(vars, 4)));
        auto bn =
            p.add_instruction(migraph::batch_norm_inference{}, conv, scale, bias, mean, variance);
        auto relu = p.add_instruction(migraph::activation{"relu"}, bn);
        p.add_instruction(migraph::pooling{"average", {1, 1}, {2, 2}, {3, 3}}, relu);
        return p;
    }
};

struct test_conv_bn_relu_pooling2
{
    static migraph::instruction_ref
    add_bn(migraph::program& p, migraph::instruction_ref x, std::size_t channels)
    {
        migraph::shape vars{migraph::shape::float_type, {channels}};
        auto scale    = p.add_literal(migraph::abs(migraph::generate_literal(vars, 1 + channels)));
        auto bias     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 2 + channels)));
        auto mean     = p.add_literal(migraph::abs(migraph::generate_literal(vars, 3 + channels)));
        auto variance = p.add_literal(migraph::abs(migraph::generate_literal(vars, 4 + channels)));
        return p.add_instruction(migraph::batch_norm_inference{}, x, scale, bias, mean, variance);
    }
    migraph::program create_program() const
    {
        migraph::program p;

        migraph::shape xs1{migraph::shape::float_type, {1, 512, 7, 7}};
        migraph::shape xs2{migraph::shape::float_type, {1, 1024, 14, 14}};
        migraph::shape ws1{migraph::shape::float_type, {2048, 512, 1, 1}};
        migraph::shape ws2{migraph::shape::float_type, {2048, 1024, 1, 1}};
        auto x1    = p.add_parameter("x1", xs1);
        auto w1    = p.add_parameter("w1", ws1);
        auto conv1 = p.add_instruction(migraph::convolution{{0, 0}, {1, 1}, {1, 1}}, x1, w1);
        auto bn1   = add_bn(p, conv1, 2048);
        auto x2    = p.add_parameter("x2", xs2);
        auto w2    = p.add_parameter("w2", ws2);
        auto conv2 = p.add_instruction(migraph::convolution{{0, 0}, {2, 2}, {1, 1}}, x2, w2);
        auto bn2   = add_bn(p, conv2, 2048);
        auto add   = p.add_instruction(migraph::add{}, bn1, bn2);
        auto relu  = p.add_instruction(migraph::activation{"relu"}, add);
        p.add_instruction(migraph::pooling{"average", {1, 1}, {2, 2}, {3, 3}}, relu);
        return p;
    }
};

int main()
{
    verify_program<test_add>();
    verify_program<test_add_broadcast>();
    verify_program<test_add_broadcast2>();
    verify_program<test_add_broadcast3>();
    verify_program<test_add_broadcast4>();
    verify_program<test_add_broadcast5>();
    verify_program<test_softmax>();
    // TODO: Add reshapes to make this a valid test case
    // verify_program<test_softmax2>();
    verify_program<test_conv>();
    verify_program<test_conv_relu>();
    verify_program<test_add_relu>();
    verify_program<test_conv_pooling>();
    verify_program<test_gemm>();
    // verify_program<test_gemm_ld>();
    verify_program<test_gemm_transposeb>();
    verify_program<test_gemm_transposea>();
    verify_program<test_gemm_transposeab>();
    verify_program<test_contiguous>();
    verify_program<test_transpose>();
    verify_program<test_batchnorm_inference>();
    verify_program<test_batchnorm_inference_2>();
    verify_program<test_conv_bn>();
    verify_program<test_conv_bn_relu_pooling>();
    verify_program<test_conv_bn_relu_pooling2>();
}
