#include <test.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/iterator_for.hpp>
#include "test_utils.hpp"
#include "test.hpp"

migraphx::program create_gelu()
{
    migraphx::program p;
    std::vector<float> data0 = {0.044715};
    std::vector<float> data1 = {0.797885};
    std::vector<float> data2 = {3};
    std::vector<float> data3 = {0.5};
    migraphx::shape s0{migraphx::shape::float_type, {1}};

    std::vector<size_t> x_dims{1, 1, 5};

    auto x         = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, x_dims});
    auto const_val = p.add_literal(migraphx::literal{s0, data0});
    auto sqrt_2_pi = p.add_literal(migraphx::literal{s0, data1});
    auto three_val = p.add_literal(migraphx::literal{s0, data2});
    auto half_val  = p.add_literal(migraphx::literal{s0, data3});

    auto mbcast_3         = p.add_instruction(migraphx::op::multibroadcast{x_dims}, three_val);
    auto pow_op           = p.add_instruction(migraphx::op::pow{}, x, mbcast_3);
    auto mbcast_const     = p.add_instruction(migraphx::op::multibroadcast{x_dims}, const_val);
    auto mul_const        = p.add_instruction(migraphx::op::mul{}, mbcast_const, pow_op);
    auto add_x            = p.add_instruction(migraphx::op::add{}, x, mul_const);
    auto mbcast_sqrt_2_pi = p.add_instruction(migraphx::op::multibroadcast{x_dims}, sqrt_2_pi);
    auto mul_add_x        = p.add_instruction(migraphx::op::mul{}, mbcast_sqrt_2_pi, add_x);
    auto tanh_op          = p.add_instruction(migraphx::op::tanh{}, mul_add_x);
    auto mbcast_half      = p.add_instruction(migraphx::op::multibroadcast{x_dims}, half_val);
    auto mul_half         = p.add_instruction(migraphx::op::mul{}, mbcast_half, tanh_op);
    auto add_mul_half     = p.add_instruction(migraphx::op::add{}, mul_half, mbcast_half);
    auto mul_x            = p.add_instruction(migraphx::op::mul{}, x, add_mul_half);
    p.add_return({mul_x});

    return p;
}

TEST_CASE(enable_fast_gelu)
{
    migraphx::program p = create_gelu();
    p.compile(migraphx::gpu::target{});
    CHECK(any_of(p, [&](auto&& i) { return i.name() == "gpu::gelu"; }));
}

TEST_CASE(disable_fast_gelu)
{
    migraphx::program p = create_gelu();
    migraphx::compile_options options;
    options.fast_math = false;
    p.compile(migraphx::gpu::target{}, options);
    CHECK(any_of(p, [&](auto&& i) { return i.name() == "gpu::gelu_new"; }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
