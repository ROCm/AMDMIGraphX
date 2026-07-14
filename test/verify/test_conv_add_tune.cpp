/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/json.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/tmp_dir.hpp>
#include <cstdlib>
#include <string>

template <migraphx::shape::type_t DType>
struct test_conv_add_tune : verify_program<test_conv_add_tune<DType>>
{
    // this test is for testing MLIR split-k convolution perfConfigs and problemKey clash in problem
    // cache
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        // choose sizes such that, it would pick mlir for convolutions
        auto x1    = mm->add_parameter("x1", {DType, {1, 256, 16, 16}});
        auto w1    = mm->add_literal(migraphx::generate_literal({DType, {1, 256, 3, 2}}, 1));
        auto x2    = mm->add_parameter("x2", {DType, {1, 256, 16, 16}});
        auto w2    = mm->add_literal(migraphx::generate_literal({DType, {1, 256, 3, 2}}, 1));
        auto conv1 = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 0}}, {"stride", {2, 2}}}),
            x1,
            w1);
        // add pooling so that it doesn't get fused with conv1.
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::average},
                                                   {"padding", {1, 1, 1, 1}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {3, 3}},
                                                   {"count_include_pad", false}}),
                                conv1);
        // conv2 + pointwise-add
        auto conv2 = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 0}}, {"stride", {2, 2}}}),
            x2,
            w2);
        mm->add_instruction(migraphx::make_op("add"), pooling, conv2);
        return p;
    }
    // Turn on Exhaustive-tune to enable split-k perf-configs from MLIR
    migraphx::compile_options get_compile_options() const
    {
        return migraphx::compile_options{.exhaustive_tune = true};
    }
    std::string section() const { return "conv"; }
};

struct test_conv_add_tune_bad_perf_config_cache
    : verify_program<test_conv_add_tune_bad_perf_config_cache>
{
    test_conv_add_tune_bad_perf_config_cache()
    {
        static migraphx::tmp_dir td{"conv_add_tune_problem_cache"};
        auto cache_path = td.path / "problem_cache.json";
        const std::string problem_config =
            "gfx1100\t48\t1\tconv -F 1 -f GNC01 -I NGC01 -O NGC01 -n 1 -c 256 -H 32 -W "
            "32 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1";
        const std::string perf_config =
            "gemm:v3:64,128,16,1,1,4,0,4,1,0,0,-1,-1,-1,-1,-1";
        migraphx::value cache_entry = migraphx::value::array{};
        cache_entry.push_back(
            {{"name", std::string{"gpu::mlir_op"}}, {"problem", problem_config}});
        cache_entry.push_back(perf_config);

        migraphx::value cache = migraphx::value::array{};
        cache.push_back(cache_entry);
        migraphx::write_string(cache_path, migraphx::to_pretty_json_string(cache));
        setenv("MIGRAPHX_PROBLEM_CACHE", cache_path.string().c_str(), 1);
    }

    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape input_shape{migraphx::shape::float_type, {1, 256, 32, 32}};
        auto x    = mm->add_parameter("x", input_shape);
        auto w    = mm->add_parameter("w", {migraphx::shape::float_type, {256, 256, 3, 3}});
        auto bias = mm->add_parameter("bias", {migraphx::shape::float_type, {256}});
        auto zero = mm->add_literal(
            migraphx::literal{{migraphx::shape::float_type, {1}}, {0.0f}});
        auto scale = mm->add_literal(
            migraphx::literal{{migraphx::shape::float_type, {1}}, {0.2f}});

        auto conv =
            mm->add_instruction(migraphx::make_op("convolution",
                                                  {{"padding", {1, 1, 1, 1}},
                                                   {"stride", {1, 1}},
                                                   {"dilation", {1, 1}}}),
                                x,
                                w);
        std::vector<std::size_t> lens = {1, 256, 32, 32};
        auto bias_bcast = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", lens}}), bias);
        auto add = mm->add_instruction(migraphx::make_op("add"), conv, bias_bcast);
        auto zero_bcast =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), zero);
        auto greater = mm->add_instruction(migraphx::make_op("greater"), add, zero_bcast);
        auto scale_bcast =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), scale);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), add, scale_bcast);
        auto cond =
            mm->add_instruction(migraphx::make_op("convert",
                                                  {{"target_type", migraphx::shape::bool_type}}),
                                greater);
        auto where = mm->add_instruction(migraphx::make_op("where"), cond, add, mul);
        auto reshape1 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 256, 16, 2, 16, 2}}}),
                                where);
        auto transpose =
            mm->add_instruction(migraphx::make_op("transpose",
                                                  {{"permutation", {5, 3, 0, 1, 2, 4}}}),
                                reshape1);
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 256, 16, 16}}}),
                            transpose);
        return p;
    }

    migraphx::compile_options get_compile_options() const
    {
        return migraphx::compile_options{};
    }

    std::string section() const { return "conv"; }
};

template struct test_conv_add_tune<migraphx::shape::float_type>;
template struct test_conv_add_tune<migraphx::shape::half_type>;
template struct test_conv_add_tune<migraphx::shape::bf16_type>;
template struct test_conv_add_tune<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_conv_add_tune<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_conv_add_tune<migraphx::shape::fp8e4m3fn_type>;
template struct test_conv_add_tune<migraphx::shape::fp8e5m2_type>;
