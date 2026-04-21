/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/eliminate_data_type_for_gpu.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/eliminate_data_type.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static void insert_miopen_pooling([[maybe_unused]] std::set<std::string>& u)
{
#if MIGRAPHX_USE_MIOPEN
    u.insert("pooling");
#endif
}

static void insert_gemm_conv(std::set<std::string>& u)
{
    u.insert("convolution");
    u.insert("quant_convolution");
    u.insert("dot");
    u.insert("quant_dot");
}

static eliminate_data_type for_device_functions()
{
    std::set<shape::type_t> unsupported_types(shape::types().begin(), shape::types().end());
    unsupported_types.erase(shape::float_type);
    unsupported_types.erase(shape::half_type);
    unsupported_types.erase(shape::bool_type);
    unsupported_types.erase(shape::int8_type);
    unsupported_types.erase(shape::uint8_type);
    unsupported_types.erase(shape::int32_type);
    unsupported_types.erase(shape::bf16_type);
    unsupported_types.erase(shape::tuple_type);

    std::set<std::string> device_functions = {
        "nonzero",
        "prefix_scan_sum",
        "rnn_var_sl_shift_output",
        "multinomial",
        "argmax",
        "argmin",
    };

    return eliminate_data_type{unsupported_types, shape::float_type, device_functions};
}

template <class F>
static auto query_device(const context* ctx, F f)
{
    return ctx != nullptr ? f(*ctx) : f();
}

static eliminate_data_type for_fp8fnuz(const context* ctx)
{
    std::set<std::string> unsupported_ops = {};

    if(not query_device(ctx, [](auto&&... args) { return hipblaslt_supported(args...); }))
    {
        unsupported_ops.insert("dot");
        unsupported_ops.insert("quant_dot");
    }

    insert_miopen_pooling(unsupported_ops);

    if(not query_device(ctx, [](auto&&... args) { return gfx_has_fp8fnuz_intrinsics(args...); }))
    {
        insert_gemm_conv(unsupported_ops);
    }
    return eliminate_data_type{
        {shape::fp8e4m3fnuz_type, shape::fp8e5m2fnuz_type}, shape::float_type, unsupported_ops};
}

static eliminate_data_type for_fp8ocp(const context* ctx)
{
    std::set<std::string> unsupported_ops = {};

    if(not query_device(ctx, [](auto&&... args) { return hipblaslt_supported(args...); }))
    {
        unsupported_ops.insert("dot");
        unsupported_ops.insert("quant_dot");
    }

    insert_miopen_pooling(unsupported_ops);

    if(not query_device(ctx, [](auto&&... args) { return gfx_has_fp8ocp_intrinsics(args...); }))
    {
        insert_gemm_conv(unsupported_ops);
    }
    return eliminate_data_type{
        {shape::fp8e4m3fn_type, shape::fp8e5m2_type}, shape::float_type, unsupported_ops};
}

static eliminate_data_type for_gemm_conv()
{
    std::set<std::string> unsupported_ops = {};
    insert_gemm_conv(unsupported_ops);

    return eliminate_data_type{{
                                   shape::bool_type,
                                   shape::uint16_type,
                                   shape::int16_type,
                                   shape::int64_type,
                                   shape::uint64_type,
                                   shape::double_type,
                               },
                               shape::float_type,
                               unsupported_ops};
}

void eliminate_data_type_for_gpu::apply(module_pass_manager& mpm) const
{
    std::set<shape::type_t> unsupported_floats;
    // No BF-16 Support on Navi21
    if(not query_device(ctx, [](auto&&... args) { return gfx_has_bf16_intrinsics(args...); }))
    {
        unsupported_floats.insert(shape::bf16_type);
    }
    if(disable_64bit)
    {
        unsupported_floats.insert(shape::double_type);
    }
    if(not unsupported_floats.empty())
        mpm.run_pass(eliminate_data_type{unsupported_floats, shape::float_type});

    if(disable_64bit)
    {
        // TODO: Check for large tensors
        mpm.run_pass(eliminate_data_type{{shape::int64_type}, shape::int32_type});
        mpm.run_pass(eliminate_data_type{{shape::uint64_type}, shape::uint32_type});
    }

    // workaround for rocBLAS unsupported error when using uint8 in quant_dot, quant_convolution &
    // pooling
    mpm.run_pass(eliminate_data_type{
        {shape::uint8_type}, shape::float_type, {"quant_convolution", "quant_dot", "pooling"}});

    mpm.run_pass(for_device_functions());

    mpm.run_pass(for_fp8fnuz(ctx));
    mpm.run_pass(for_fp8ocp(ctx));

    mpm.run_pass(for_gemm_conv());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
