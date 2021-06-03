#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include <migraphx/verify.hpp>

TEST_CASE(dequantizelinear_uint8)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {0, 1, 2, 5, 10, 50, 100, 150, 250};
    migraphx::shape ss{migraphx::shape::float_type, {1}};
    std::vector<float> sv = {2};
    migraphx::shape zs{migraphx::shape::uint8_type, {1}};
    std::vector<uint8_t> zv = {0};
    auto create_program     = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        auto z   = mm->add_literal(zs, zv);
        mm->add_instruction(migraphx::make_op("dequantizelinear", {{"axis", 1}}), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::ref::target{});
    auto result = p1.eval({}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 2, 4, 10, 20, 100, 200, 300, 500};
    EXPECT(results_vector == gold);
}

TEST_CASE(dequantizelinear_int8)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {-128, -100, -50, -1, 0, 1, 50, 100, 127};
    migraphx::shape ss{migraphx::shape::float_type, {1}};
    std::vector<float> sv = {2};
    migraphx::shape zs{migraphx::shape::int8_type, {1}};
    std::vector<int8_t> zv = {0};
    auto create_program    = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        auto z   = mm->add_literal(zs, zv);
        mm->add_instruction(migraphx::make_op("dequantizelinear", {{"axis", 1}}), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::ref::target{});
    auto result = p1.eval({}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-256, -200, -100, -2, 0, 2, 100, 200, 254};
    EXPECT(results_vector == gold);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
