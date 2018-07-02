
#include <rtg/program.hpp>
#include <rtg/operators.hpp>
#include <rtg/generate.hpp>
#include <rtg/cpu/cpu_target.hpp>
#include <rtg/miopen/miopen_target.hpp>
#include <rtg/miopen/miopen.hpp>
#include <rtg/miopen/hip.hpp>
#include <rtg/manage_ptr.hpp>

#include <miopen/miopen.h>

#include "test.hpp"
#include "verify.hpp"

template <class V>
rtg::argument run_cpu()
{
    V v;
    auto p = v.create_program();
    p.compile(rtg::cpu::cpu_target{});
    return p.eval(v.create_params());
}

template <class V>
rtg::argument run_gpu()
{
    V v;
    auto p = v.create_program();
    p.compile(rtg::miopen::miopen_target{});

    auto m = v.create_params();
    for(auto&& e : m)
    {
        e.second = rtg::miopen::to_gpu(e.second);
    }

    m["output"] = rtg::miopen::to_gpu(rtg::generate_argument(p.get_parameter_shape("output")));

    return rtg::miopen::from_gpu(p.eval(m));
}

template <class V>
void verify_program()
{
    auto cpu_arg = run_cpu<V>();
    auto gpu_arg = run_gpu<V>();
    visit_all(cpu_arg, gpu_arg)([](auto cpu, auto gpu) { EXPECT(test::verify_range(cpu, gpu)); });
}

struct test_add
{
    rtg::program create_program() const
    {
        rtg::program p;
        rtg::shape s{rtg::shape::float_type, {3}};
        auto x = p.add_parameter("x", s);
        auto y = p.add_parameter("y", s);
        p.add_instruction(rtg::add{}, x, y);
        return p;
    }

    rtg::program::parameter_map create_params() const
    {
        rtg::program::parameter_map m;
        m["x"] = rtg::generate_argument({rtg::shape::float_type, {3}});
        m["y"] = rtg::generate_argument({rtg::shape::float_type, {3}});
        return m;
    }
};

struct test_add_broadcast
{
    rtg::program create_program() const
    {
        rtg::program p;
        rtg::shape s{rtg::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {rtg::shape::float_type, {2, 2, 3}});
        auto y  = p.add_parameter("y", {rtg::shape::float_type, {2, 2}});
        auto by = p.add_instruction(rtg::broadcast{0}, x, y);
        p.add_instruction(rtg::add{}, x, by);
        return p;
    }

    rtg::program::parameter_map create_params() const
    {
        rtg::program::parameter_map m;
        m["x"] = rtg::generate_argument({rtg::shape::float_type, {2, 2, 3}});
        m["y"] = rtg::generate_argument({rtg::shape::float_type, {2, 2}});
        return m;
    }
};

struct test_conv_relu
{
    rtg::program create_program() const
    {
        rtg::program p;
        auto input   = p.add_parameter("x", rtg::shape{rtg::shape::float_type, {4, 3, 3, 3}});
        auto weights = p.add_parameter("w", rtg::shape{rtg::shape::float_type, {4, 3, 3, 3}});
        auto conv    = p.add_instruction(rtg::convolution{}, input, weights);
        p.add_instruction(rtg::activation{"relu"}, conv);
        return p;
    }

    rtg::program::parameter_map create_params() const
    {
        rtg::program::parameter_map m;
        m["x"] = rtg::generate_argument({rtg::shape::float_type, {4, 3, 3, 3}});
        m["w"] = rtg::generate_argument({rtg::shape::float_type, {4, 3, 3, 3}});
        return m;
    }
};

struct test_conv_pooling
{
    rtg::program create_program() const
    {
        rtg::program p;
        auto input   = p.add_parameter("x", rtg::shape{rtg::shape::float_type, {4, 3, 32, 32}});
        auto weights = p.add_parameter("w", rtg::shape{rtg::shape::float_type, {4, 3, 3, 3}});
        auto conv    = p.add_instruction(rtg::convolution{}, input, weights);
        auto pooling = p.add_instruction(rtg::pooling{"max"}, conv);
        p.add_instruction(rtg::activation{"relu"}, pooling);
        return p;
    }

    rtg::program::parameter_map create_params() const
    {
        rtg::program::parameter_map m;
        m["x"] = rtg::generate_argument({rtg::shape::float_type, {4, 3, 32, 32}});
        m["w"] = rtg::generate_argument({rtg::shape::float_type, {4, 3, 3, 3}});
        return m;
    }
};

struct test_gemm
{
    rtg::program create_program() const
    {
        rtg::program p;
        auto a = p.add_parameter("a", rtg::shape{rtg::shape::float_type, {4, 5}});
        auto b = p.add_parameter("b", rtg::shape{rtg::shape::float_type, {5, 3}});
        p.add_instruction(rtg::gemm{}, a, b);
        return p;
    }

    rtg::program::parameter_map create_params() const
    {
        rtg::program::parameter_map m;
        m["a"] = rtg::generate_argument({rtg::shape::float_type, {4, 5}});
        m["b"] = rtg::generate_argument({rtg::shape::float_type, {5, 3}});
        return m;
    }
};

int main()
{
    verify_program<test_add>();
    verify_program<test_add_broadcast>();
    verify_program<test_conv_relu>();
    verify_program<test_conv_pooling>();
    verify_program<test_gemm>();
}
