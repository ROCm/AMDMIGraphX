#include <migraphx/remove_qdq_pairs.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>

#include <migraphx/serialize.hpp>

#include <migraphx/verify.hpp>

bool is_quantizelinear(migraphx::instruction& ins) { return ins.name() == "quantizelinear"; }
bool is_dequantizelinear(migraphx::instruction& ins) { return ins.name() == "dequantizelinear"; }

TEST_CASE(quantizelinear)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {-300, 200, 129, 1, 2, 3, 500, 1000, 50};
    std::vector<float> xv2 = {1, 2, 3, -300, 200, 129, 500, 1000, 50};
    migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    migraphx::shape zs{migraphx::shape::int8_type, {1}};
    std::vector<int8_t> zv = {0};
    auto create_program   = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto x2  = mm->add_literal(xs, xv2);
        auto s   = mm->add_literal(ss, sv);
        auto z   = mm->add_literal(zs, zv);
        auto qx  = mm->add_instruction(migraphx::make_op("quantizelinear"), x2, s, z);
        auto dqx = mm->add_instruction(migraphx::make_op("dequantizelinear"), qx, s, z);
        auto q   = mm->add_instruction(migraphx::make_op("quantizelinear"), x, s, z);
        auto dq  = mm->add_instruction(migraphx::make_op("dequantizelinear"), q, s, z);
        auto d   = mm->add_instruction(migraphx::make_op("dot"), dq, dqx);
        auto q1  = mm->add_instruction(migraphx::make_op("quantizelinear"), d, s, z);
        auto dq1 = mm->add_instruction(migraphx::make_op("dequantizelinear"), q1, s, z);
        auto su  = mm->add_instruction(migraphx::make_op("sub"), dq1, x);
        auto q2  = mm->add_instruction(migraphx::make_op("quantizelinear"), su, s, z);
        auto dq2 = mm->add_instruction(migraphx::make_op("dequantizelinear"), q2, s, z);
        auto a   = mm->add_instruction(migraphx::make_op("add"), dq2, dq1);
        mm->add_return({a});
        return p;
    };

    migraphx::program p1 = migraphx::parse_onnx("/home/amt/holding/quant1/mobilenetv2-7-opt.quant.onnx");//
    //migraphx::program p1 = create_program();
    migraphx::program p2 = migraphx::parse_onnx("/home/amt/holding/quant1/mobilenetv2-7-opt.quant.onnx");//
    //migraphx::program p2 = create_program();
    std::cout << "Original: " << std::endl;
    p2.debug_print();

    migraphx::remove_qdq_pairs opt;
    opt.apply(*p2.get_main_module());
    std::cout << "Pass applied: " << std::endl;
    p2.debug_print();
    EXPECT(any_of(*p1.get_main_module(), &is_quantizelinear));
    EXPECT(none_of(*p2.get_main_module(), &is_quantizelinear));
    EXPECT(any_of(*p1.get_main_module(), &is_dequantizelinear));
    EXPECT(none_of(*p2.get_main_module(), &is_dequantizelinear));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
