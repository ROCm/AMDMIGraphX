#include <iostream>
#include <vector>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
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

TEST_CASE(gpu_target_copy)
{
    migraphx::target gpu_t = migraphx::gpu::target{};
    migraphx::target cpu_t = migraphx::cpu::target{};
    migraphx::shape s{migraphx::shape::int8_type, {2, 3, 4, 5}};

    auto cpu_arg_orig  = migraphx::generate_argument(s, 0x123456L);
    auto gpu_arg       = gpu_t.copy_to(cpu_arg_orig);
    auto cpu_arg_final = gpu_t.copy_from(gpu_arg);

    std::vector<int8_t> val_orig;
    cpu_arg_orig.visit([&](auto v) { val_orig.assign(v.begin(), v.end()); });
    std::vector<int8_t> val_final;
    cpu_arg_final.visit([&](auto v) { val_final.assign(v.begin(), v.end()); });

    EXPECT(migraphx::verify_range(val_orig, val_final));
}

TEST_CASE(int8_quantization)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::program::parameter_map& m_in,
                       std::vector<float>& res) {
        std::vector<migraphx::program::parameter_map> cali_data;
        cali_data.push_back(m_in);
        migraphx::quantize_int8(p, t, cali_data);
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

        auto result = t.copy_from(p.eval(m).back());
        result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
    };

    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);
        p.add_instruction(migraphx::op::dot{}, pa, pb, pc);

        return p;
    };

    {
        auto p = create_program();
        migraphx::program::parameter_map m;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        m["a"] = migraphx::generate_argument(sa);
        m["b"] = migraphx::generate_argument(sb);
        m["c"] = migraphx::generate_argument(sc);
        std::vector<float> cpu_result;
        migraphx::target cpu_t = migraphx::cpu::target{};
        run_prog(p, cpu_t, m, cpu_result);

        std::vector<float> gpu_result;
        migraphx::target gpu_t = migraphx::gpu::target{};
        run_prog(p, gpu_t, m, gpu_result);

        EXPECT(migraphx::verify_range(cpu_result, gpu_result));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
