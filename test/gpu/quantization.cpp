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
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/onnx.hpp>
#include <test.hpp>
#include <migraphx/half.hpp>
#include <migraphx/make_op.hpp>

TEST_CASE(gpu_target_copy)
{
    migraphx::target gpu_t = migraphx::make_target("gpu");
    migraphx::target ref_t = migraphx::make_target("ref");
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

TEST_CASE(int8_quantization_bug)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::parameter_map& m_in,
                       std::vector<float>& res) {
        std::vector<migraphx::parameter_map> cali_data;
        cali_data.push_back(m_in);
        // migraphx::quantize_int8(p, t, cali_data);
        p.compile(t);
        // std::cout << p << std::endl;
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

    size_t m_dim = 5;
    size_t k_dim = 16;
    size_t n_dim = 8;

    auto create_program = [&] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {m_dim, k_dim}};
        migraphx::shape sb{migraphx::shape::float_type, {k_dim, n_dim}};
        migraphx::shape sc{migraphx::shape::float_type, {m_dim, n_dim}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);

        // quantizelinear for arg0
        migraphx::shape ss1{migraphx::shape::int8_type, {m_dim, k_dim}};
        auto literal1 = mm->add_literal(0.00738189f);
        auto bcast1   = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", ss1.lens()}}), literal1);
        auto quant_linear1 = mm->add_instruction(migraphx::make_op("quantizelinear"), pa, bcast1);

        // quantizelinear for arg1
        migraphx::shape ss2{migraphx::shape::int8_type, {k_dim, n_dim}};
        auto literal2 = mm->add_literal(0.00787402f);
        auto bcast2   = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", ss2.lens()}}), literal2);
        auto quant_linear2 = mm->add_instruction(migraphx::make_op("quantizelinear"), pb, bcast2);

        auto dot = mm->add_instruction(migraphx::op::quant_dot{}, quant_linear1, quant_linear2);

        migraphx::shape ss{migraphx::shape::float_type, {m_dim, n_dim}};
        auto literal = mm->add_literal(5.81251188e-05f);
        auto bcast   = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", ss.lens()}}), literal);
        auto dequant = mm->add_instruction(migraphx::make_op("dequantizelinear"), dot, bcast);
        mm->add_return({dequant});

        return p;
    };

    {
        auto p = create_program();
        migraphx::parameter_map m;
        migraphx::shape sa{migraphx::shape::float_type, {m_dim, k_dim}};
        migraphx::shape sb{migraphx::shape::float_type, {k_dim, n_dim}};
        migraphx::shape sc{migraphx::shape::float_type, {m_dim, n_dim}};

        m["a"] = migraphx::generate_argument(sa);
        m["b"] = migraphx::generate_argument(sb);

        std::vector<float> b_vals;
        m["b"].visit([&](auto v) { b_vals.assign(v.begin(), v.end()); });
        for(auto&& val : b_vals)
        {
            std::cout << val << ", ";
        }
        std::cout << std::endl;
        std::vector<float> ref_result;
        migraphx::target ref_t = migraphx::make_target("ref");
        run_prog(p, ref_t, m, ref_result);
        // print ref_result
        std::cout << "ref_result: ";
        for(auto&& v : ref_result)
            std::cout << v << " ";
        std::cout << std::endl;

        std::vector<float> gpu_result;
        migraphx::target gpu_t = migraphx::make_target("gpu");
        run_prog(p, gpu_t, m, gpu_result);
        std::cout << "gpu_result: ";
        for(auto&& v : gpu_result)
            std::cout << v << " ";
        std::cout << std::endl;

        EXPECT(migraphx::verify_range(ref_result, gpu_result));
    }
}

TEST_CASE(int8_dot)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::parameter_map& m_in,
                       std::vector<float>& res) {
        std::vector<migraphx::parameter_map> cali_data;
        cali_data.push_back(m_in);
        // migraphx::quantize_int8(p, t, cali_data);
        p.compile(t);
        // std::cout << p << std::endl;
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

    size_t m_dim = 5;
    size_t k_dim = 16;
    size_t n_dim = 8;

    auto create_program = [&] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::int8_type, {m_dim, k_dim}};
        migraphx::shape sb{migraphx::shape::int8_type, {k_dim, n_dim}};
        // migraphx::shape sc{migraphx::shape::int32_type, {m_dim, n_dim}};
        auto pa = mm->add_parameter("a", sa);
        auto pb = mm->add_parameter("b", sb);

        auto dot = mm->add_instruction(migraphx::op::quant_dot{}, pa, pb);

        mm->add_return({dot});

        return p;
    };

    {
        auto p = create_program();
        migraphx::parameter_map m;
        migraphx::shape sa{migraphx::shape::int8_type, {m_dim, k_dim}};
        migraphx::shape sb{migraphx::shape::int8_type, {k_dim, n_dim}};
        // migraphx::shape sc{migraphx::shape::float_type, {m_dim, n_dim}};

        std::vector<int8_t> a = {59,   102,  25,   34,   34,  127,  -102, 51,  25,   119, -42, 59,
                                 0,    93,   42,   127,  127, -110, 127,  85,  -127, 93,  -34, -76,
                                 -102, -76,  25,   34,   119, -102, 51,   -85, 102,  51,  -34, 51,
                                 -8,   -119, -59,  -76,  85,  -51,  -127, -68, 51,   -25, -59, 102,
                                 -51,  25,   -68,  85,   -8,  0,    76,   -34, 85,   17,  0,   34,
                                 110,  25,   -102, -102, -17, 8,    -68,  25,  -8,   93,  51,  -127,
                                 59,   76,   119,  42,   25,  -17,  -8,   -119};
        std::vector<int8_t> b = {
            56,  95,   24,   32,  32,   119,  -95, 48,   24,  111, -40,  56,  0,   87,   40,  119,
            119, -103, 119,  79,  -119, 87,   -32, -71,  -95, -71, 24,   32,  111, -95,  48,  -79,
            95,  48,   -32,  48,  -8,   -111, -56, -71,  79,  -48, -119, -64, 48,  -24,  -56, 95,
            -48, 24,   -64,  79,  -8,   0,    71,  -32,  79,  16,  0,    32,  103, 24,   -95, -95,
            -16, 8,    -64,  24,  -8,   87,   48,  -119, 56,  71,  111,  40,  24,  -16,  -8,  -111,
            71,  8,    95,   64,  79,   103,  119, -79,  -32, -8,  79,   8,   40,  -48,  24,  -32,
            24,  -103, 71,   87,  95,   -24,  111, 8,    48,  -16, 0,    103, 71,  -103, 0,   103,
            64,  87,   -119, 111, -24,  -87,  40,  -111, 8,   -40, -127, -64, 56,  0,    -16, -111};

        m["a"] = migraphx::argument{sa, a.data()};
        m["b"] = migraphx::argument{sb, b.data()};
        std::vector<float> ref_result;
        migraphx::target ref_t = migraphx::make_target("ref");
        run_prog(p, ref_t, m, ref_result);
        // print ref_result
        std::cout << "ref_result: ";
        for(auto&& v : ref_result)
            std::cout << v << " ";
        std::cout << std::endl;

        std::vector<float> gpu_result;
        migraphx::target gpu_t = migraphx::make_target("gpu");
        run_prog(p, gpu_t, m, gpu_result);
        std::cout << "gpu_result: ";
        for(auto&& v : gpu_result)
            std::cout << v << " ";
        std::cout << std::endl;

        EXPECT(migraphx::verify_range(ref_result, gpu_result));
    }
}

TEST_CASE(div_test)
{
    auto run_prog = [](migraphx::program p,
                       const migraphx::target& t,
                       migraphx::parameter_map& m_in,
                       std::vector<float>& res) {
        std::vector<migraphx::parameter_map> cali_data;
        cali_data.push_back(m_in);
        // migraphx::quantize_int8(p, t, cali_data);
        p.compile(t);
        // std::cout << p << std::endl;
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

    auto create_program = [&] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sa{migraphx::shape::float_type, {1}};
        // migraphx::shape sb{migraphx::shape::float_type, {1}};
        // migraphx::shape sc{migraphx::shape::int32_type, {m_dim, n_dim}};
        auto pa = mm->add_parameter("a", sa);
        // auto pb = mm->add_parameter("b", sb);
        auto pb = mm->add_literal(0.00787402f);

        auto div = mm->add_instruction(migraphx::op::div{}, pa, pb);

        mm->add_return({div});

        return p;
    };

    {
        auto p = create_program();
        migraphx::parameter_map m;
        migraphx::shape sa{migraphx::shape::float_type, {1}};
        migraphx::shape sb{migraphx::shape::float_type, {1}};
        // migraphx::shape sc{migraphx::shape::float_type, {m_dim, n_dim}};

        std::vector<float> a = {0.5f};
        std::vector<float> b = {0.00787402f};

        m["a"] = migraphx::argument{sa, a.data()};
        // m["b"] = migraphx::argument{sb, b.data()};
        std::vector<float> ref_result;
        migraphx::target ref_t = migraphx::make_target("ref");
        run_prog(p, ref_t, m, ref_result);
        // print ref_result
        std::cout << "ref_result: ";
        for(auto&& v : ref_result)
            std::cout << v << " ";
        std::cout << std::endl;

        std::vector<float> gpu_result;
        migraphx::target gpu_t = migraphx::make_target("gpu");
        run_prog(p, gpu_t, m, gpu_result);
        std::cout << "gpu_result: ";
        for(auto&& v : gpu_result)
            std::cout << v << " ";
        std::cout << std::endl;

        EXPECT(migraphx::verify_range(ref_result, gpu_result));
    }
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
        migraphx::target ref_t = migraphx::make_target("ref");
        run_prog(p, ref_t, m, ref_result);

        std::vector<float> gpu_result;
        migraphx::target gpu_t = migraphx::make_target("gpu");
        run_prog(p, gpu_t, m, gpu_result);

        // Note: the tolerance for mlir_enabled result is temporarily bumped
        // higher because the lowering pipeline between mlir fallback and
        // regular non-mlir pipeline diverged. MLIR fallback uses the
        // rewrite_quantization at the very end of the pipeline, whereas
        // the regular pipeline uses the rewrite_quantization in the much
        // earlier stage.
        if(migraphx::gpu::mlir_enabled())
            EXPECT(migraphx::verify_range(ref_result, gpu_result, 1e5));
        else
            EXPECT(migraphx::verify_range(ref_result, gpu_result));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
