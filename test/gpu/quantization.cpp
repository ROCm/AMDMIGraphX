/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <iostream>
#include <vector>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/onnx.hpp>
#include <test.hpp>
#include <migraphx/half.hpp>

TEST_CASE(gpu_target_copy)
{
    migraphx::target gpu_t = migraphx::gpu::target{};
    migraphx::target ref_t = migraphx::ref::target{};
    migraphx::shape s{migraphx::shape::int8_type, {2, 3, 4, 5}};

    auto ref_arg_orig  = migraphx::generate_argument(s, 0x123456L);
    auto gpu_arg       = gpu_t.copy_to(ref_arg_orig);
    auto ref_arg_final = gpu_t.copy_from(gpu_arg);

    std::vector<int8_t> val_orig;
    ref_arg_orig.visit([&](auto v) { val_orig.assign(v.begin(), v.end()); });
    std::vector<int8_t> val_final;
    ref_arg_final.visit([&](auto v) { val_final.assign(v.begin(), v.end()); });

    EXPECT(migraphx::verify_range(val_orig, val_final));
}

TEST_CASE(int8_quantization)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::parameter_map& m_in,
                       std::vector<float>& res) {
        std::vector<migraphx::parameter_map> cali_data;
        cali_data.push_back(m_in);
        migraphx::quantize_int8(p, t, cali_data);
        p.compile(t);
        migraphx::parameter_map m;
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
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {5, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {5, 8}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);
        mm->add_instruction(migraphx::op::dot{}, pa, pb);

        return p;
    };

    {
        auto p = create_program();
        migraphx::parameter_map m;
        migraphx::shape sa{migraphx::shape::float_type, {5, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {5, 8}};
        m["a"] = migraphx::generate_argument(sa);
        m["b"] = migraphx::generate_argument(sb);
        std::vector<float> ref_result;
        migraphx::target ref_t = migraphx::ref::target{};
        run_prog(p, ref_t, m, ref_result);

        std::vector<float> gpu_result;
        migraphx::target gpu_t = migraphx::gpu::target{};
        run_prog(p, gpu_t, m, gpu_result);

        EXPECT(migraphx::verify_range(ref_result, gpu_result));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
