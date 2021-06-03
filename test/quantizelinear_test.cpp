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

TEST_CASE(quantizelinear_uint8)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {-300, 200, 129, 0, 1, 2, 500, 1000, 50};
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
        mm->add_instruction(migraphx::make_op("quantizelinear", {{"axis", 1}}), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::ref::target{});
    auto result = p1.eval({}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 100, 65, 0, 1, 1, 250, 255, 25};
    EXPECT(results_vector == gold);
}

TEST_CASE(quantizelinear_int8)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {-300, -200, -129, 0, 1, 2, 50, 500, 1000};
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
        mm->add_instruction(migraphx::make_op("quantizelinear", {{"axis", 1}}), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    p1.compile(migraphx::ref::target{});
    auto result = p1.eval({}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-128, -100, -65, 0, 1, 1, 25, 127, 127};
    EXPECT(results_vector == gold);
}

TEST_CASE(quantizelinear_axes)
{
    {
        /* Axis 0*/
        migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> xv = {
            -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
        migraphx::shape ss{migraphx::shape::float_type, {2}};
        std::vector<float> sv = {2, 4};
        migraphx::shape zs{migraphx::shape::uint8_type, {2}};
        std::vector<uint8_t> zv = {0, 0};
        auto create_program     = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            auto z   = mm->add_literal(zs, zv);
            mm->add_instruction(migraphx::make_op("quantizelinear", {{"axis", 0}}), x, s, z);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::ref::target{});
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(18);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0, 255, 65, 0, 2, 2, 0, 255, 255, 0, 150, 32, 0, 1, 1, 0, 150, 138};
        EXPECT(results_vector == gold);
    }

    {
        /* Axis 1*/
        migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> xv = {
            -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
        migraphx::shape ss{migraphx::shape::float_type, {3}};
        std::vector<float> sv = {2, 4, 6};
        migraphx::shape zs{migraphx::shape::uint8_type, {3}};
        std::vector<uint8_t> zv = {0, 0, 0};
        auto create_program     = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            auto z   = mm->add_literal(zs, zv);
            mm->add_instruction(migraphx::make_op("quantizelinear", {{"axis", 1}}), x, s, z);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::ref::target{});
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(18);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0, 255, 65, 0, 1, 1, 0, 100, 92, 0, 255, 65, 0, 1, 1, 0, 100, 92};
        EXPECT(results_vector == gold);
    }

    {
        /* Axis 2*/
        migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> xv = {
            -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
        migraphx::shape ss{migraphx::shape::float_type, {3}};
        std::vector<float> sv = {2, 4, 6};
        migraphx::shape zs{migraphx::shape::uint8_type, {3}};
        std::vector<uint8_t> zv = {0, 0, 0};
        auto create_program     = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            auto z   = mm->add_literal(zs, zv);
            mm->add_instruction(migraphx::make_op("quantizelinear", {{"axis", 2}}), x, s, z);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::ref::target{});
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(18);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0, 150, 22, 0, 1, 1, 0, 150, 92, 0, 150, 22, 0, 1, 1, 0, 150, 92};
        EXPECT(results_vector == gold);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
