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

bool is_quantizelinear(migraphx::instruction& ins) { return ins.name() == "quantizelinear"; }
bool is_dequantizelinear(migraphx::instruction& ins) { return ins.name() == "dequantizelinear"; }

TEST_CASE(quantizelinear)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {-300, 200, 129, 1, 2, 3, 500, 1000, 50};
    migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    auto create_program   = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        mm->add_instruction(migraphx::make_op("quantizelinear"), x, s);
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    migraphx::rewrite_quantization opt;
    opt.apply(*p2.get_main_module());
    EXPECT(any_of(*p1.get_main_module(), &is_quantizelinear));
    EXPECT(none_of(*p2.get_main_module(), &is_quantizelinear));
}

TEST_CASE(dequantizelinear)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {0, 1, 2, 5, 10, 50, 100, 150, 250};
    migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    migraphx::shape zs{migraphx::shape::uint8_type, {1, 3, 3}};
    std::vector<uint8_t> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto create_program     = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        auto z   = mm->add_literal(zs, zv);
        mm->add_instruction(migraphx::make_op("dequantizelinear"), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    migraphx::rewrite_quantization opt;
    opt.apply(*p2.get_main_module());
    EXPECT(any_of(*p1.get_main_module(), &is_dequantizelinear));
    EXPECT(none_of(*p2.get_main_module(), &is_dequantizelinear));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
