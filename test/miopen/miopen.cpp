
#include <migraph/program.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
#include <migraph/cpu/cpu_target.hpp>
#include <migraph/miopen/miopen_target.hpp>
#include <migraph/miopen/miopen.hpp>
#include <migraph/miopen/hip.hpp>
#include <migraph/manage_ptr.hpp>

#include <miopen/miopen.h>

#include "test.hpp"
#include "verify.hpp"

template <class V>
migraph::argument run_cpu()
{
    V v;
    auto p = v.create_program();
    p.compile(migraph::cpu::cpu_target{});
    return p.eval(v.create_params());
}

template <class V>
migraph::argument run_gpu()
{
    V v;
    auto p = v.create_program();
    p.compile(migraph::miopen::miopen_target{});

    auto m = v.create_params();
    for(auto&& e : m)
    {
        e.second = migraph::miopen::to_gpu(e.second);
    }

    m["output"] =
        migraph::miopen::to_gpu(migraph::generate_argument(p.get_parameter_shape("output")));

    return migraph::miopen::from_gpu(p.eval(m));
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
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {3}};
        auto x = p.add_parameter("x", s);
        auto y = p.add_parameter("y", s);
        p.add_instruction(migraph::add{}, x, y);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {3}});
        m["y"] = migraph::generate_argument({migraph::shape::float_type, {3}});
        return m;
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

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {2, 2, 3}});
        m["y"] = migraph::generate_argument({migraph::shape::float_type, {2, 2}});
        return m;
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

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {4, 3, 3, 3}});
        m["w"] = migraph::generate_argument({migraph::shape::float_type, {4, 3, 3, 3}});
        return m;
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

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {4, 3, 32, 32}});
        m["w"] = migraph::generate_argument({migraph::shape::float_type, {4, 3, 3, 3}});
        return m;
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

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["a"] = migraph::generate_argument({migraph::shape::float_type, {4, 5}});
        m["b"] = migraph::generate_argument({migraph::shape::float_type, {5, 3}});
        return m;
    }
};

struct test_contiguous
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {32, 16, 128, 128}, {262144, 128, 1, 16384}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraph::contiguous{}, x);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {32, 16, 128, 128}});
        return m;
    }
};

void contiguous_test()
{
    migraph::shape a_shape{migraph::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);

    migraph::program p;
    auto l = p.add_literal(migraph::literal{a_shape, data});
    p.add_instruction(migraph::contiguous{}, l);
    p.compile(migraph::miopen::miopen_target{});
    auto result = p.eval({});

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<size_t> new_lens    = {1, 3, 2, 2};
    std::vector<size_t> new_strides = {12, 1, 6, 3};
    std::vector<float> gold         = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    EXPECT(test::verify_range(results_vector, gold));
}

int main()
{
    verify_program<test_add>();
    verify_program<test_add_broadcast>();
    verify_program<test_conv_relu>();
    verify_program<test_conv_pooling>();
    verify_program<test_gemm>();
    // verify_program<test_contiguous>();
    contiguous_test();
}
