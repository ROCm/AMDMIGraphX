#include <iostream>
#include <vector>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"
#include <migraphx/half.hpp>

TEST_CASE(target_copy)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::program::parameter_map& m_in,
                       std::vector<float>& res) {
        p.compile(t);
        migraphx::program::parameter_map m;
        for(auto&& x : p.get_parameter_shapes())
        {
            if(m_in.count(x.first) > 0)
            {
                m[x.first] = t.copy_to(m_in[x.first]);
            }
            else
            {
                m[x.first] = t.allocate(x.second);
            }
        }

        auto result = t.copy_from(p.eval(m));
        result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
    };

    auto create_program = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto p1 = p.add_parameter("x", s);
        auto p2 = p.add_parameter("y", s);
        p.add_instruction(migraphx::op::add{}, p1, p2);

        return p;
    };

    {
        auto p = create_program();
        migraphx::program::parameter_map m;
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        m["x"] = migraphx::generate_argument(s);
        std::vector<float> cpu_result;
        migraphx::target cpu_t = migraphx::cpu::target{};
        run_prog(p, cpu_t, m, cpu_result);

        std::vector<float> gpu_result;
        migraphx::target gpu_t = migraphx::gpu::target{};
        run_prog(p, gpu_t, m, gpu_result);

        EXPECT(migraphx::verify_range(cpu_result, gpu_result));
    }
}

TEST_CASE(dot_large_alpha_beta_float)
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);

        p.add_instruction(migraphx::op::dot{20.0f, 50.5f}, pa, pb, pc);

        return p;
    };

    auto create_int8_quantized_prog = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);
        // quantize parameter a to int8 type, multiply the scale
        std::vector<float> vfa(sa.elements(), 0.1f);
        auto fa = p.add_literal(migraphx::literal(sa, vfa));
        auto ma = p.add_instruction(migraphx::op::mul{}, fa, pa);
        // add the shift
        std::vector<float> vsa(sa.elements(), 1.0f);
        auto sfta = p.add_literal(migraphx::literal(sa, vsa));
        auto msa = p.add_instruction(migraphx::op::add{}, sfta, ma);
        auto ra = p.add_instruction(migraphx::op::round{}, msa);
        auto ca = p.add_instruction(migraphx::op::clip{127.0f, -128.0f}, ra);
        auto qa = p.add_instruction(migraphx::op::convert{migraphx::shape::int8_type}, ca);

        // quantize parameter b to int8 type
        auto insert_loc = std::next(pb);
        std::vector<float> vfb(sb.elements(), 0.1f);
        auto fb = p.add_literal(migraphx::literal(sb, vfb));
        auto mb = p.insert_instruction(insert_loc, migraphx::op::mul{}, fb, pb);
        auto rb = p.insert_instruction(insert_loc, migraphx::op::round{}, mb);
        auto cb = p.insert_instruction(insert_loc, migraphx::op::clip{127.0f, -128.0f}, rb);
        auto qb = p.insert_instruction(insert_loc, migraphx::op::convert{migraphx::shape::int8_type}, cb);

        // quantize parameter b to int32 type
        auto qc = p.insert_instruction(std::next(pc), migraphx::op::convert{migraphx::shape::int32_type}, pc);

        auto qdot = p.add_instruction(migraphx::op::quant_dot{2000, 51}, qa, qb, qc);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, qdot);

        return p;
    };

    auto p = create_program();

    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    // default scale 64.0f is used for all args
    migraphx::quantize_int8(p, {"dot"}, quant_params);

    auto qp = create_int8_quantized_prog();

    EXPECT(p == qp);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
