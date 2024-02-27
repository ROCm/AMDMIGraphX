/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "run_verify.hpp"
#include <migraphx/ranges.hpp>
#include <test.hpp>

#ifdef HAVE_GPU
#include <migraphx/gpu/analyze_streams.hpp>
#include <migraphx/gpu/target.hpp>
#endif
#ifdef HAVE_CPU
#include <migraphx/cpu/target.hpp>
#endif

inline void check_gpu_streams(const migraphx::program& p)
{
#ifdef HAVE_GPU
    const auto* mm = p.get_main_module();
    auto races     = migraphx::gpu::analyze_streams(*mm);
    for(auto&& race : races)
    {
        std::cout << "FAILED: " << std::endl;
        std::cout << "Race condition detected for: ";
        mm->debug_print(race.ins);
        std::cout << "Should happen after: ";
        mm->debug_print(race.before);
    }
#else
    (void)p;
#endif
}

void validate_gpu(const migraphx::program& p, const migraphx::parameter_map& m)
{
    check_gpu_streams(p);

    // Ensure the program doesn't modify the context in a dry run
    auto ctx = p.get_context();
    assert(&ctx != &p.get_context());
    EXPECT(is_shared(ctx, p.get_context()));
    p.dry_run(m);
    EXPECT(is_shared(ctx, p.get_context()));
}

int main(int argc, const char* argv[])
{
    run_verify rv;
    rv.add_validation_for("gpu", &validate_gpu);
    rv.disable_test_for("cpu", {
        "test_if_lp", "test_if_param", "test_if_literal", "test_select_module_add",
            "test_select_module_reduce", "test_select_module_conv", "test_split_single_dyn_dim",
            "test_instancenorm_large_3d<migraphx::shape::float_type>",
            "test_instancenorm_large_3d<migraphx::shape::half_type>",
        // these tests are disabled due issue of lossy downcast, see issue#2517
#if defined(__GNUC__) and !defined(__clang__)
            "batch_quant_dot_1<migraphx::fp8::float8<migraphx::fp8::f8_type::fp8, true>, float>",
            "quant_dot_3args_4<migraphx::fp8::float8<migraphx::fp8::f8_type::fp8, true>, float>",
            "quant_dot_3args_5<migraphx::fp8::float8<migraphx::fp8::f8_type::fp8, true>, float>",
#else
                "batch_quant_dot_1<migraphx::fp8::fp8e4m3fnuz, float>",
                "quant_dot_3args_4<migraphx::fp8::fp8e4m3fnuz, float>",
                "quant_dot_3args_5<migraphx::fp8::fp8e4m3fnuz, float>",
#endif
            "test_block_reduce_small<3, migraphx::shape::int8_type>",
            "test_block_reduce_small<4, migraphx::shape::int8_type>",
            "test_block_reduce_small<8, migraphx::shape::int8_type>",
            "test_block_reduce_small<16, migraphx::shape::int8_type>",
            "test_block_reduce_small<25, migraphx::shape::int8_type>",
            "test_block_reduce_small<32, migraphx::shape::int8_type>",
            "test_block_reduce_small<64, migraphx::shape::int8_type>",
            "test_block_reduce_small<67, migraphx::shape::int8_type>",
            "test_block_reduce_small<128, migraphx::shape::int8_type>",
            "test_block_reduce_small<129, migraphx::shape::int8_type>",
    });
    rv.disable_test_for("gpu",
                        {// These passes on MI300 but fails on others, same issue as CPU.
                         "batch_quant_dot_1<migraphx::fp8::fp8e4m3fnuz, float>",
                         "quant_dot_3args_4<migraphx::fp8::fp8e4m3fnuz, float>",
                         "quant_dot_3args_5<migraphx::fp8::fp8e4m3fnuz, float>"});
    rv.run(argc, argv);
}
