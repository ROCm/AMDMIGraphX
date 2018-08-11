
#include <migraph/program.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
#include <migraph/cpu/cpu_target.hpp>
#include <migraph/gpu/target.hpp>
#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/type_name.hpp>

#include <miopen/miopen.h>

#include "test.hpp"
#include "verify.hpp"

template <class V>
migraph::argument run_cpu()
{
    V v;
    auto p = v.create_program();
    p.compile(migraph::cpu::cpu_target{});
    migraph::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraph::generate_argument(x.second);
    }
    return p.eval(m);
}

template <class V>
migraph::argument run_gpu()
{
    V v;
    auto p = v.create_program();
    p.compile(migraph::gpu::target{});

    migraph::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraph::gpu::to_gpu(migraph::generate_argument(x.second));
    }

    return migraph::gpu::from_gpu(p.eval(m));
}

template <class V>
void verify_program()
{
    auto cpu_arg = run_cpu<V>();
    auto gpu_arg = run_gpu<V>();
    visit_all(cpu_arg, gpu_arg)([](auto cpu, auto gpu) {
        if(not test::verify_range(cpu, gpu))
        {
            std::cout << "FAILED: " << migraph::get_type_name<V>() << std::endl;
        }
    });
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
        auto mean     = p.add_parameter("mean", vars);
        auto variance = p.add_parameter("variance", vars);
        auto scale    = p.add_parameter("scale", vars);
        auto bias     = p.add_parameter("bias", vars);
        p.add_instruction(migraph::batch_norm_inference{}, x, mean, variance, scale, bias);
        return p;
    }
};

void batch_norm_inference_test()
{
    migraph::program p;
    const size_t width = 2, height = 2, channels = 4, batches = 2;
    const float x_val = 8.0f, mean_val = 2.0f, variance_val = 4.0f, scale_val = 2.0f,
                bias_val   = 1.0f;
    const float output_val = scale_val * (x_val - mean_val) / (std::sqrt(variance_val)) + bias_val;

    migraph::shape s{migraph::shape::float_type, {batches, channels, height, width}};
    migraph::shape vars{migraph::shape::float_type, {channels}};
    std::vector<float> x_data(width * height * channels * batches);
    std::vector<float> scale_data(channels);
    std::vector<float> bias_data(channels);
    std::vector<float> mean_data(channels);
    std::vector<float> variance_data(channels);

    std::fill(x_data.begin(), x_data.end(), x_val);
    std::fill(mean_data.begin(), mean_data.end(), mean_val);
    std::fill(variance_data.begin(), variance_data.end(), variance_val);
    std::fill(scale_data.begin(), scale_data.end(), scale_val);
    std::fill(bias_data.begin(), bias_data.end(), bias_val);

    auto x        = p.add_literal(migraph::literal{s, x_data});
    auto scale    = p.add_literal(migraph::literal{vars, scale_data});
    auto bias     = p.add_literal(migraph::literal{vars, bias_data});
    auto mean     = p.add_literal(migraph::literal{vars, mean_data});
    auto variance = p.add_literal(migraph::literal{vars, variance_data});

    p.add_instruction(migraph::batch_norm_inference{}, x, mean, variance, scale, bias);
    p.compile(migraph::gpu::target{});

    migraph::program::parameter_map m;
    m["output"] = migraph::gpu::to_gpu(migraph::generate_argument(p.get_parameter_shape("output")));
    auto result = migraph::gpu::from_gpu(p.eval(m));

    std::vector<float> result_vector(width * height * channels * batches);
    std::vector<float> gold(width * height * channels * batches);
    std::fill(gold.begin(), gold.end(), output_val);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    EXPECT(test::verify_range(result_vector, gold));
}

int main()
{
    verify_program<test_add>();
    verify_program<test_add_broadcast>();
    verify_program<test_conv_relu>();
    verify_program<test_conv_pooling>();
    verify_program<test_gemm>();
    // verify_program<test_gemm_ld>();
    verify_program<test_gemm_transposeb>();
    verify_program<test_gemm_transposea>();
    verify_program<test_gemm_transposeab>();
    verify_program<test_contiguous>();
    verify_program<test_transpose>();
    verify_program<test_batchnorm_inference>();
    batch_norm_inference_test();
}
